"""Shared document text extraction utilities.

Used by both the dashboard upload endpoint (routes/memories.py) and the
SDK upload endpoint (routes/sdk_memories.py) so extraction logic stays
in one place.
"""

import io
import logging
from typing import Optional

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

ACCEPTED_UPLOAD_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json",
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

NO_VISION_DETAIL = (
    "The configured model cannot read images, so this image was not uploaded. "
    "Set SYNAP_LLM_MODEL_VISION to a model that reads images, then restart the server."
)

IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024       # 10 MB


async def extract_text(
    filename: str,
    content: bytes,
    *,
    client_id: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> str:
    """Extract plain text from a supported file.

    `client_id` / `instance_id` attribute an image read's model usage; without
    them the call is still made and traced, but not billed to anyone.

    Dispatches by extension:
      - Images  → the vision reader (ingestion.multimodal.describe_image)
      - PDF     → pypdf
      - DOCX    → python-docx
      - XLSX    → openpyxl
      - PPTX    → python-pptx
      - All else → UTF-8 decode
    """
    name = filename.lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""

    try:
        if ext in IMAGE_EXTENSIONS:
            # The same reader conversation images use, so an image read here
            # and one attached to a turn are read by one model, one prompt,
            # and billed and traced the same way.
            from synap.cloud.application.ingestion.multimodal import describe_image, has_vision_model

            if not has_vision_model():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=NO_VISION_DETAIL)
            return await describe_image(
                content, IMAGE_MEDIA_TYPES[ext], filename=filename,
                client_id=client_id, instance_id=instance_id,
                operation="upload.image_extract",
            )

        if name.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

        if name.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(content))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    parts.append("\t".join(c.text.strip() for c in row.cells if c.text.strip()))
            return "\n".join(parts)

        if name.endswith(".xlsx"):
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
            parts = []
            for sheet in wb.worksheets:
                parts.append(f"[Sheet: {sheet.title}]")
                for row in sheet.iter_rows(values_only=True):
                    row_text = "\t".join(str(c) for c in row if c is not None)
                    if row_text.strip():
                        parts.append(row_text)
            return "\n".join(parts)

        if name.endswith(".pptx"):
            import pptx
            prs = pptx.Presentation(io.BytesIO(content))
            parts = []
            for i, slide in enumerate(prs.slides, 1):
                parts.append(f"[Slide {i}]")
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            text = para.text.strip()
                            if text:
                                parts.append(text)
                if slide.has_notes_slide:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        parts.append(f"[Notes] {notes}")
            return "\n".join(parts)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract text from {filename}: {exc}",
        )

    return content.decode("utf-8", errors="replace")
