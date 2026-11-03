"""Download the models Core needs, once, and check every file against a pinned hash.

    python -m synap.boot.first_start

Run by the compose file before the server starts, into the `models` volume (HF_HOME). It
fetches two things, straight from their publishers, and nothing else:

  * the embedding model, at one exact revision, as the files a Hugging Face cache holds
  * the spaCy English model the rule-based splitter needs, as a wheel unpacked into
    ``<models>/spacy`` (put that folder on PYTHONPATH)

It uses plain HTTPS, not the Hugging Face library, so no usage report is sent anywhere.
A file whose hash differs from the pinned one is deleted and the run fails; there is no
"latest". A finished run leaves a marker file, and the next start does nothing.

Before downloading anything it checks that the developer has chosen a model (``check_llm_settings``):
Core ships no default, and a placeholder left in ``.env`` is refused by name.
"""
from __future__ import annotations

import hashlib
import http.client
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

EMBEDDING_REPO = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBEDDING_REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"
# path -> (hash kind, hash). Large files are LFS objects with a sha256; small ones are plain git
# blobs, whose sha1 is over "blob <size>\0" + content. Both come from the Hub's file listing at
# the revision above, not from the files themselves.
EMBEDDING_FILES = {
    "config.json": ("git-sha1", "91a1faf6a9239179d70a70347628b182fa6a0cfb"),
    "config_sentence_transformers.json": ("git-sha1", "b11fb81a2a2b0e6ce5d44aabf2a71bf8c804eaab"),
    "modules.json": ("git-sha1", "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
    "1_Pooling/config.json": ("git-sha1", "553a16bda12e2a6d2bb35de78c6ea264b7856e6a"),
    "special_tokens_map.json": ("git-sha1", "b1879d702821e753ffe4245048eee415d54a9385"),
    "tokenizer_config.json": ("git-sha1", "bda426376d27170ae0a14e7d4cf4087efdc39eb5"),
    "sentencepiece.bpe.model": ("sha256", "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "tokenizer.json": ("sha256", "39feb9863a378165ab9c5c689047203d789422966c0c58721c5309fd039a8edc"),
    "model.safetensors": ("sha256", "21bf1a120b1c6562aeec379dfa9039b0d360591c784cb1c6786e87256b738ee1"),
}
EMBEDDING_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"

# The second model: synap-er (entity resolution) matches names with all-MiniLM-L6-v2, which it loads by
# name. Without this it was fetched at the first memory, unpinned and over the internet, into a folder the
# server may not be allowed to write to. Laid out the same way as the model above, so the loader finds it
# by name in the cache with no network (HF_HUB_OFFLINE=1). Hashes from the Hub's listing at this revision.
ENTITY_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ENTITY_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENTITY_MODEL_FILES = {
    "1_Pooling/config.json": ("git-sha1", "d1514c3162bbe87b343f565fadc62e6c06f04f03"),
    "config.json": ("git-sha1", "72b987fd805cfa2b58c4c8c952b274a11bfd5a00"),
    "config_sentence_transformers.json": ("git-sha1", "fd1b291129c607e5d49799f87cb219b27f98acdf"),
    "modules.json": ("git-sha1", "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
    "sentence_bert_config.json": ("git-sha1", "59d594003bf59880a884c574bf88ef7555bb0202"),
    "special_tokens_map.json": ("git-sha1", "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
    "tokenizer.json": ("git-sha1", "cb202bfe2e3c98645018a6d12f182a434c9d3e02"),
    "tokenizer_config.json": ("git-sha1", "c79f2b6a0cea6f4b564fed1938984bace9d30ff0"),
    "vocab.txt": ("git-sha1", "fb140275c155a9c7c5a3b3e0e77a9e839594a938"),
    "model.safetensors": ("sha256", "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"),
}

SPACY_WHEEL = "en_core_web_sm-3.7.1-py3-none-any.whl"
SPACY_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/" + SPACY_WHEEL
SPACY_SHA256 = "86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889"

MARKER = ".first-start-done"
_CHUNK = 1 << 20


class FirstStartError(Exception):
    pass


@dataclass(frozen=True)
class Source:
    """Where the files come from. Tests point these at a local server."""
    embedding_url: str = EMBEDDING_URL
    spacy_url: str = SPACY_URL
    entity_model_url: str = EMBEDDING_URL


def digest(path: Path, kind: str) -> str:
    if kind == "sha256":
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(_CHUNK):
                h.update(chunk)
        return h.hexdigest()
    if kind == "git-sha1":
        data = path.read_bytes()
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's own object id
    raise FirstStartError(f"unknown hash kind {kind}")


ATTEMPTS = 8
PROGRESS_SECONDS = 10
# A connection that sends almost nothing is as useless as a dead one. If fewer than STALL_MIN_BYTES arrive
# in any STALL_SECONDS, the attempt is dropped and the next one resumes from the part already on disk.
STALL_SECONDS = 60
STALL_MIN_BYTES = 600_000
_READ = 65_536


def _size(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def _fetch(url: str, dest: Path, *, attempts: int = ATTEMPTS, log=print) -> None:
    """Download to ``dest.part`` and resume it if a run was cut off.

    Says how far it is every ``PROGRESS_SECONDS``, so a slow download does not look like a hang.
    A connection that drops is retried with a growing wait (about two minutes in all), and a
    run that still fails keeps the part it has: starting again continues from there. An attempt
    that added bytes to the part does not count against ``attempts``, so a poor link that keeps
    cutting off but keeps moving still finishes; only attempts that get nowhere run out.
    """
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    n = 0
    while n < attempts:
        n += 1
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resumed = bool(have) and r.status == 206
                done = have if resumed else 0
                length = r.headers.get("Content-Length")
                total = done + int(length) if length and length.isdigit() else 0
                if total > 5_000_000:
                    log(f"first-start: {dest.name}: {_size(total)}" + (f", continuing from {_size(done)}" if done else ""))
                started = last_report = window_start = time.monotonic()
                start_done = window_done = done
                with part.open("ab" if resumed else "wb") as f:
                    # read1 returns what has arrived; read(n) would wait for all n bytes, and a
                    # connection that trickles never fills them, so the part on disk would not grow.
                    while chunk := r.read1(_READ):
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - window_start >= STALL_SECONDS:
                            if done - window_done < STALL_MIN_BYTES:
                                raise TimeoutError(f"stalled: {_size(done - window_done)} in {STALL_SECONDS} s")
                            window_start, window_done = now, done
                        if total > 5_000_000 and now - last_report >= PROGRESS_SECONDS:
                            last_report = now
                            rate = (done - start_done) / max(now - started, 1e-9)
                            log(f"first-start: {dest.name}: {_size(done)} of {_size(total)} "
                                f"({100 * done // total}%), {rate / 1e6:.1f} MB/s")
            if total and done != total:
                raise ConnectionError(f"the connection closed early, at {_size(done)} of {_size(total)}")
            part.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            last = e
            log(f"first-start: {dest.name}: attempt {n} of {attempts} failed ({type(e).__name__}: {e})")
            if (part.stat().st_size if part.exists() else 0) > have:
                n = 0  # it moved: this was not a dead attempt
                time.sleep(5)
            elif n < attempts:
                time.sleep(min(5 * n, 30))
    raise FirstStartError(
        f"could not download {dest.name}: {last}. Check your internet connection, then start again: "
        "the download continues from where it stopped."
    )


def _verified(path: Path, kind: str, want: str) -> bool:
    return path.is_file() and digest(path, kind) == want


def _fetch_hub_model(cache: Path, url: str, repo: str, revision: str, files: dict, log) -> Path:
    """Lay the files out as a Hugging Face cache does, so the loader finds them by name + revision.

    ``cache`` is the folder the loader looks in: the model volume itself for the embedding model (the
    server loads it with ``cache_folder`` set to it), and its ``hub/`` folder for a model loaded by name
    with no cache setting, which is where the Hugging Face library looks under HF_HOME."""
    snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    for rel, (kind, want) in files.items():
        target = snap / rel
        if _verified(target, kind, want):
            continue
        log(f"first-start: downloading {rel}")
        _fetch(url.format(repo=repo, revision=revision, path=rel), target, log=log)
        if not _verified(target, kind, want):
            target.unlink(missing_ok=True)
            raise FirstStartError(f"{rel} does not match the pinned hash; it was deleted. Nothing was loaded.")
    refs = snap.parent.parent / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)
    return snap


def fetch_embedding_model(models: Path, source: Source, log=print) -> Path:
    return _fetch_hub_model(models, source.embedding_url, EMBEDDING_REPO, EMBEDDING_REVISION, EMBEDDING_FILES, log)


def fetch_entity_model(models: Path, source: Source, log=print) -> Path:
    return _fetch_hub_model(models / "hub", source.entity_model_url, ENTITY_MODEL_REPO, ENTITY_MODEL_REVISION,
                            ENTITY_MODEL_FILES, log)


def fetch_spacy_model(models: Path, source: Source, log=print) -> Path:
    out = models / "spacy"
    if (out / "en_core_web_sm").is_dir():
        return out
    wheel = models / "downloads" / SPACY_WHEEL
    if not (wheel.is_file() and digest(wheel, "sha256") == SPACY_SHA256):
        log(f"first-start: downloading {SPACY_WHEEL}")
        _fetch(source.spacy_url, wheel, log=log)
        if digest(wheel, "sha256") != SPACY_SHA256:
            wheel.unlink(missing_ok=True)
            raise FirstStartError(f"{SPACY_WHEEL} does not match the pinned hash; it was deleted.")
    tmp = models / "spacy.tmp"
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            # a wheel is a zip: refuse any entry that would land outside the target folder
            if name.startswith("/") or ".." in Path(name).parts:
                raise FirstStartError(f"unsafe path in {SPACY_WHEEL}: {name}")
        z.extractall(tmp)
    tmp.replace(out)
    wheel.unlink(missing_ok=True)
    return out


def marker_text() -> str:
    """What the marker file holds when every pinned download is in place."""
    return f"{EMBEDDING_REVISION} {ENTITY_MODEL_REVISION} {SPACY_SHA256}"


def is_done(models: Path) -> bool:
    """True once ``run`` has finished for the pinned models and spaCy wheel (the marker matches)."""
    marker = models / MARKER
    return marker.is_file() and marker.read_text().strip() == marker_text()


def run(models: Path, source: Source | None = None, log=print) -> None:
    source = source or Source()
    if is_done(models):
        log("first-start: models already in place")
        return
    models.mkdir(parents=True, exist_ok=True)
    fetch_embedding_model(models, source, log)
    fetch_entity_model(models, source, log)
    fetch_spacy_model(models, source, log)
    (models / MARKER).write_text(marker_text() + "\n")
    log("first-start: done")


def check_llm_settings() -> None:
    """Core has no default model. Say so now, before a 2.3 GB download, not at the first ingest."""
    from synap.cloud.infrastructure.llm import settings as llm_settings
    try:
        configured = llm_settings.load()
    except llm_settings.LLMSettingsError as e:
        raise FirstStartError(str(e)) from None
    if configured is None:
        raise FirstStartError(
            f"Core has no default model: set {llm_settings.PROVIDER_VAR} and {llm_settings.MODEL_VAR} in your .env file "
            "(see .env.example)")


def main() -> int:
    models = Path(os.environ.get("HF_HOME") or "/models")
    try:
        check_llm_settings()
        run(models)
    except FirstStartError as e:
        print(f"first-start: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
