"""Images in conversations: the vision reader and the parts renderer.

A conversation turn can now carry images next to its text. The memory pipeline
reads text only, so each image is read once by a vision model and the turn is
rendered to ONE string before anything else sees it:

    text part        -> its text
    image, read      -> [image: <filename>]\\n<description>\\n[/image]
    image, not read  -> [image: <filename> (could not be read)]

joined with "\\n". Three surfaces use this: `POST /api/v1/memories/create-multipart`,
REST `POST /v1/conversations/messages` and the gRPC conversation event. The file
upload routes use `describe_image` alone, through `document_extractor`.

The image bytes are never stored: not in the document, not in metadata, not in
the LLM call trace (the manager redacts inline images before a trace is built).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The image prompt is read the same way for an upload and for an image in a
# conversation. It lives in the prompts package so Core can ship its own text.
from synap.cloud.application.ingestion.prompts.image_description import VISION_PROMPT

logger = logging.getLogger(__name__)

IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})

MAX_IMAGE_BYTES = 10 * 1024 * 1024          # per image, as received
MAX_IMAGES_PER_CALL = 10
MAX_IMAGE_BYTES_FOR_API = 4 * 1024 * 1024   # resize above this before the model
# Refused before decoding. A 144M-pixel PNG fits in 10 MB and decodes to ~1 GB.
MAX_IMAGE_PIXELS = 40_000_000
# Per image. Under the SDK's 30 s read timeout, so a slow model makes a failed
# read and a response rather than a client-side timeout and a retry.
VISION_TIMEOUT_SECONDS = 20.0
# What a client is told about an image that was not read. The cause (a provider
# error, a timeout) is logged here and never returned: it can carry provider
# text the client has no business seeing.
READ_FAILED_ERROR = "could not be read"

# The SDK puts a gRPC event's parts under this metadata key (the proto has no
# field for them). The server takes it out before the metadata goes anywhere.
CONTENT_PARTS_METADATA_KEY = "synap_content_parts"

ONLY_IMAGES_DETAIL = (
    "Only image attachments (png, jpeg, webp) are supported in memories.create "
    "today; use memories.create_from_file for documents."
)

DEFAULT_IMAGE_FILENAME = "image"


class ImagePartError(ValueError):
    """An attached image the request cannot use. Carries the HTTP status."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class ImageInput:
    data: bytes
    media_type: str
    filename: Optional[str] = None


@dataclass
class ImageRead:
    filename: str
    media_type: str
    description: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.description is not None

    def attachment(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "media_type": self.media_type,
            "status": "read" if self.ok else "failed",
            "error": self.error,
        }


# ── validation ──────────────────────────────────────────────────────────────

def sniff_image_media_type(data: bytes) -> Optional[str]:
    """The image type the bytes actually are, from their magic number."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image(data: bytes, declared_media_type: Optional[str]) -> str:
    """Check one image and return its real media type.

    The bytes decide, not the declared type: a caller who passed raw bytes with
    no filename declares application/octet-stream for a perfectly good PNG, and
    a file named .png can hold anything. Raises ImagePartError (413 / 415).
    """
    if len(data) > MAX_IMAGE_BYTES:
        raise ImagePartError(413, "Each image must be 10 MB or smaller.")
    sniffed = sniff_image_media_type(data)
    if sniffed is None:
        raise ImagePartError(415, ONLY_IMAGES_DETAIL)
    if declared_media_type and declared_media_type.lower() != sniffed:
        logger.info(
            "image_media_type_corrected declared=%s sniffed=%s",
            declared_media_type, sniffed,
        )
    return sniffed


def decode_base64_image(data: str) -> bytes:
    """Inline image data from a JSON part. Raises ImagePartError (400 / 413)."""
    # Cheap bound before decoding: base64 is 4 chars per 3 bytes.
    if len(data) > (MAX_IMAGE_BYTES * 4) // 3 + 4:
        raise ImagePartError(413, "Each image must be 10 MB or smaller.")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ImagePartError(400, "Image `data` must be base64.")


# ── reading ─────────────────────────────────────────────────────────────────

_CACHE_SIZE = 256
# Keyed by tenant as well as by the bytes: two clients sending the same image
# never share an entry.
_description_cache: "OrderedDict[Tuple[str, str, str, str], str]" = OrderedDict()


def _vision_model() -> str:
    from synap.cloud.infrastructure.llm import settings as llm_settings
    configured = llm_settings.active()
    if configured:  # one provider and model for everything; it must be able to read images
        return configured.model_for("VISION")
    return os.environ.get("SYNAP_VISION_MODEL", "gpt-4o")


def _vision_provider():
    from synap.cloud.infrastructure.llm import settings as llm_settings
    from synap.cloud.infrastructure.llm.models import LLMProvider
    configured = llm_settings.active()
    return configured.provider if configured else LLMProvider.OPENAI


# Names that tell a model reads images. Only used when one model is configured for
# everything and a local or self-hosted server is behind it: a hosted OpenAI or
# Anthropic model is taken to read images. A developer whose model is not matched
# here names a vision model for the step with SYNAP_LLM_MODEL_VISION.
_VISION_MODEL_MARKERS = (
    "llava", "vision", "-vl", "vl:", "minicpm-v", "pixtral", "moondream", "gemma3", "gpt-4o",
    "gpt-4.1", "gpt-5", "gpt-6", "claude", "qwen2.5vl", "llama4",
)


def has_vision_model() -> bool:
    """Whether a model that can read images is configured for the vision step.

    With the legacy settings (no ``SYNAP_LLM_PROVIDER``) the step is OpenAI with
    ``SYNAP_VISION_MODEL`` (default ``gpt-4o``), which always names a vision model.
    With one provider and model for everything, a hosted OpenAI or Anthropic model
    counts; for Ollama and openai-compatible servers the vision-step model (the
    ``SYNAP_LLM_MODEL_VISION`` setting, else the default model) must carry a name
    that says it reads images.
    """
    from synap.cloud.infrastructure.llm import settings as llm_settings
    from synap.cloud.infrastructure.llm.models import LLMProvider

    configured = llm_settings.active()
    if configured is None:
        return True
    if configured.base_url is None and configured.provider in (LLMProvider.OPENAI, LLMProvider.ANTHROPIC):
        return True
    if "VISION" in configured.step_models:
        return True
    model = configured.model_for("VISION").lower()
    return any(marker in model for marker in _VISION_MODEL_MARKERS)


def warn_if_no_vision_model() -> bool:
    """Log one startup warning when no vision model is configured.

    Returns True when it warned. Logs nothing when a vision model is configured,
    which is always so in Cloud with the legacy settings.
    """
    if has_vision_model():
        return False
    logger.warning(
        "No vision model is configured: images cannot be read. A conversation turn with an "
        "image is still recorded, with '[image: <name> (could not be read)]' in place of the "
        "description, and an image upload is refused. Set SYNAP_LLM_MODEL_VISION to a model "
        "that reads images to turn this on."
    )
    return True


class ImageTooLarge(ValueError):
    """More pixels than MAX_IMAGE_PIXELS. Reported as a failed read."""


def _resize_for_vision(data: bytes, media_type: str) -> Tuple[bytes, str]:
    """Resize to <= 2048px and re-encode as JPEG when over the API limit.

    Blocking (PIL): call it through `asyncio.to_thread`. The pixel count is
    read from the header before anything is decoded.
    """
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    width, height = img.size
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageTooLarge(f"{width}x{height} exceeds {MAX_IMAGE_PIXELS} pixels")
    if len(data) <= MAX_IMAGE_BYTES_FOR_API:
        return data, media_type
    img = img.convert("RGB")
    img.thumbnail((2048, 2048), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


async def describe_image(
    data: bytes,
    media_type: str,
    *,
    filename: Optional[str] = None,
    client_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    operation: str = "ingestion.vision",
) -> str:
    """The text in an image plus a description of it, from a vision model.

    Goes through the LLM manager so the call is privacy-guarded, traced and
    billed like every other model call. Cached by the image's sha256, so an
    image sent twice (a retry, a replayed event) is paid for once.
    """
    model = _vision_model()
    key = (str(client_id or ""), str(instance_id or ""), hashlib.sha256(data).hexdigest(), model)
    cached = _description_cache.get(key)
    if cached is not None:
        _description_cache.move_to_end(key)
        return cached

    from synap.cloud.infrastructure.llm import get_llm_manager

    image_bytes, send_type = await asyncio.to_thread(_resize_for_vision, data, media_type)
    b64 = base64.b64encode(image_bytes).decode()
    response = await get_llm_manager().complete(
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{send_type};base64,{b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
        force_provider=_vision_provider(),
        force_model=model,
        allow_cross_provider_fallback=False,
        step="VISION",
        max_tokens=2000,
        client_id=client_id,
        instance_id=instance_id,
        operation=operation,
    )
    description = (response.content or "").strip()
    if description:
        _description_cache[key] = description
        if len(_description_cache) > _CACHE_SIZE:
            _description_cache.popitem(last=False)
    logger.info(
        "vision_image_read filename=%s media_type=%s bytes=%d chars=%d",
        filename, media_type, len(data), len(description),
    )
    return description


async def read_images(
    images: Sequence[ImageInput],
    *,
    client_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    operation: str = "ingestion.vision",
) -> List[ImageRead]:
    """Read every image concurrently. One failing never fails the others."""

    async def _one(image: ImageInput) -> ImageRead:
        name = image.filename or DEFAULT_IMAGE_FILENAME
        try:
            description = await asyncio.wait_for(
                describe_image(
                    image.data, image.media_type, filename=name,
                    client_id=client_id, instance_id=instance_id, operation=operation,
                ),
                timeout=VISION_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 -- reported per image, never raised
            logger.warning(
                "vision_image_read_failed filename=%s error_type=%s error=%s",
                name, type(exc).__name__, exc,
            )
            return ImageRead(name, image.media_type, error=READ_FAILED_ERROR)
        if not description:
            logger.warning("vision_image_read_failed filename=%s error_type=empty_answer", name)
            return ImageRead(name, image.media_type, error=READ_FAILED_ERROR)
        return ImageRead(name, image.media_type, description=description)

    return list(await asyncio.gather(*(_one(i) for i in images)))


# ── rendering ───────────────────────────────────────────────────────────────

def render_parts(parts: Sequence[Any]) -> str:
    """One string for a list of text parts (str) and ImageRead results."""
    out: List[str] = []
    for part in parts:
        if isinstance(part, ImageRead):
            name = part.filename or DEFAULT_IMAGE_FILENAME
            if part.ok:
                out.append(f"[image: {name}]\n{part.description}\n[/image]")
            else:
                out.append(f"[image: {name} (could not be read)]")
        else:
            out.append(str(part))
    return "\n".join(out)


async def read_and_render(
    parts: Sequence[Any],
    *,
    client_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    operation: str = "ingestion.vision",
) -> Tuple[str, List[ImageRead]]:
    """Read the ImageInputs in `parts` (str and ImageInput, in order) and render.

    Returns the rendered string and one ImageRead per image, in order.
    """
    images = [p for p in parts if isinstance(p, ImageInput)]
    reads = iter(await read_images(
        images, client_id=client_id, instance_id=instance_id, operation=operation,
    ))
    resolved = [next(reads) if isinstance(p, ImageInput) else p for p in parts]
    return render_parts(resolved), [p for p in resolved if isinstance(p, ImageRead)]


def parts_from_wire(raw_parts: Any, *, strict: bool = True) -> List[Any]:
    """Text and inline-base64 image wire parts -> str and ImageInput, in order.

    `strict` raises ImagePartError on a bad image (the REST route answers the
    caller with it). Not strict, a bad image becomes a failed read instead: the
    gRPC path has already acknowledged the event and has nobody to tell.
    Malformed structure (not a list, unknown type) always raises ValueError.
    """
    if not isinstance(raw_parts, list):
        raise ValueError("content parts must be a list")
    out: List[Any] = []
    images = 0
    for part in raw_parts:
        if not isinstance(part, dict):
            raise ValueError("each content part must be an object")
        kind = part.get("type")
        if kind == "text":
            out.append(str(part.get("text") or ""))
        elif kind == "image":
            images += 1
            if images > MAX_IMAGES_PER_CALL:
                raise ImagePartError(400, f"At most {MAX_IMAGES_PER_CALL} images per message.")
            filename = part.get("filename") or None
            declared = part.get("media_type")
            try:
                data = decode_base64_image(str(part.get("data") or ""))
                media_type = validate_image(data, declared)
            except ImagePartError as exc:
                if strict:
                    raise
                out.append(ImageRead(
                    filename or DEFAULT_IMAGE_FILENAME, str(declared or ""), error=exc.detail,
                ))
                continue
            out.append(ImageInput(data=data, media_type=media_type, filename=filename))
        else:
            raise ValueError(f"unknown content part type {kind!r}")
    return out
