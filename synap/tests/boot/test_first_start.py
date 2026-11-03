"""First-start downloader: pinned, verified, resumable, and quiet when done."""
from __future__ import annotations

import hashlib
import io
import re
import subprocess
import threading
import zipfile
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from synap.boot import first_start as fs


def _git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _wheel(extra: dict | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("en_core_web_sm/__init__.py", "x = 1\n")
        for k, v in (extra or {}).items():
            z.writestr(k, v)
    return buf.getvalue()


class Origin:
    """A local stand-in for Hugging Face and GitHub that counts requests and honours Range."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.hits = files, []
        origin = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                origin.hits.append(self.path)
                body = origin.files.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                start = int(re.match(r"bytes=(\d+)-", rng).group(1)) if rng else 0
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(body) - start))
                self.end_headers()
                self.wfile.write(body[start:])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def world(monkeypatch):
    small, big = b'{"a": 1}\n', b"B" * 300_000
    wheel = _wheel()
    files = {"config.json": small, "model.safetensors": big}
    monkeypatch.setattr(fs, "EMBEDDING_FILES", {
        "config.json": ("git-sha1", _git_sha1(small)),
        "model.safetensors": ("sha256", hashlib.sha256(big).hexdigest()),
    })
    monkeypatch.setattr(fs, "SPACY_SHA256", hashlib.sha256(wheel).hexdigest())
    served = {f"/{fs.EMBEDDING_REPO}/resolve/{fs.EMBEDDING_REVISION}/{k}": v for k, v in files.items()}
    entity = {"config.json": small, "model.safetensors": b"E" * 50_000}
    monkeypatch.setattr(fs, "ENTITY_MODEL_FILES", {
        "config.json": ("git-sha1", _git_sha1(small)),
        "model.safetensors": ("sha256", hashlib.sha256(entity["model.safetensors"]).hexdigest()),
    })
    served.update({f"/{fs.ENTITY_MODEL_REPO}/resolve/{fs.ENTITY_MODEL_REVISION}/{k}": v for k, v in entity.items()})
    served[f"/{fs.SPACY_WHEEL}"] = wheel
    origin = Origin(served)
    source = fs.Source(
        embedding_url=origin.base + "/{repo}/resolve/{revision}/{path}",
        spacy_url=origin.base + f"/{fs.SPACY_WHEEL}",
        entity_model_url=origin.base + "/{repo}/resolve/{revision}/{path}",
    )
    yield origin, source, served
    origin.close()


def test_the_second_model_is_pinned_and_laid_out_like_a_hub_cache(world, tmp_path):
    """MiniLM (synap-er's name matcher) is fetched here, so nothing is downloaded at the first memory."""
    origin, source, _ = world
    fs.run(tmp_path, source, log=lambda *_: None)
    snap = tmp_path / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / fs.ENTITY_MODEL_REVISION
    assert (snap / "model.safetensors").stat().st_size == 50_000
    assert (snap.parent.parent / "refs" / "main").read_text() == fs.ENTITY_MODEL_REVISION
    assert fs.is_done(tmp_path)


def test_the_real_pin_of_the_second_model_is_exact():
    import re as _re
    assert _re.fullmatch(r"[0-9a-f]{40}", fs.ENTITY_MODEL_REVISION)
    assert "model.safetensors" in fs.ENTITY_MODEL_FILES and "modules.json" in fs.ENTITY_MODEL_FILES
    for kind, h in fs.ENTITY_MODEL_FILES.values():
        assert (kind == "sha256" and len(h) == 64) or (kind == "git-sha1" and len(h) == 40)


def test_pins_are_exact():
    assert re.fullmatch(r"[0-9a-f]{40}", fs.EMBEDDING_REVISION)
    assert "main" not in fs.EMBEDDING_REVISION and "latest" not in fs.EMBEDDING_REVISION
    assert re.fullmatch(r"[0-9a-f]{64}", fs.SPACY_SHA256)
    for kind, h in fs.EMBEDDING_FILES.values():
        assert (kind == "sha256" and len(h) == 64) or (kind == "git-sha1" and len(h) == 40)
    assert "model.safetensors" in fs.EMBEDDING_FILES


def test_git_digest_matches_git(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"hello\n")
    want = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True).stdout.strip()
    assert fs.digest(p, "git-sha1") == want


def test_downloads_verifies_and_lays_out_a_hub_cache(world, tmp_path):
    origin, source, _ = world
    fs.run(tmp_path, source, log=lambda *_: None)
    snap = tmp_path / "models--Snowflake--snowflake-arctic-embed-l-v2.0" / "snapshots" / fs.EMBEDDING_REVISION
    assert (snap / "config.json").is_file() and (snap / "model.safetensors").stat().st_size == 300_000
    assert (snap.parent.parent / "refs" / "main").read_text() == fs.EMBEDDING_REVISION
    assert (tmp_path / "spacy" / "en_core_web_sm" / "__init__.py").is_file()
    assert (tmp_path / fs.MARKER).is_file()


def test_is_done_is_false_until_a_run_finishes_and_false_for_another_revision(world, tmp_path):
    origin, source, _ = world
    assert fs.is_done(tmp_path) is False
    fs.run(tmp_path, source, log=lambda *_: None)
    assert fs.is_done(tmp_path) is True
    (tmp_path / fs.MARKER).write_text("another-revision another-hash\n")
    assert fs.is_done(tmp_path) is False


def test_second_run_makes_no_requests(world, tmp_path):
    origin, source, _ = world
    fs.run(tmp_path, source, log=lambda *_: None)
    n = len(origin.hits)
    fs.run(tmp_path, source, log=lambda *_: None)
    assert len(origin.hits) == n


def test_a_wrong_file_is_deleted_and_nothing_is_marked_done(world, tmp_path):
    origin, source, served = world
    key = f"/{fs.EMBEDDING_REPO}/resolve/{fs.EMBEDDING_REVISION}/model.safetensors"
    origin.files[key] = b"X" * 300_000
    with pytest.raises(fs.FirstStartError, match="pinned hash"):
        fs.run(tmp_path, source, log=lambda *_: None)
    snap = tmp_path / "models--Snowflake--snowflake-arctic-embed-l-v2.0" / "snapshots" / fs.EMBEDDING_REVISION
    assert not (snap / "model.safetensors").exists()
    assert not (tmp_path / fs.MARKER).exists()


def test_a_cut_off_download_resumes_instead_of_starting_over(world, tmp_path):
    origin, source, served = world
    key = f"/{fs.EMBEDDING_REPO}/resolve/{fs.EMBEDDING_REVISION}/model.safetensors"
    snap = tmp_path / "models--Snowflake--snowflake-arctic-embed-l-v2.0" / "snapshots" / fs.EMBEDDING_REVISION
    snap.mkdir(parents=True)
    (snap / "model.safetensors.part").write_bytes(served[key][:100_000])
    fs.run(tmp_path, source, log=lambda *_: None)
    assert (snap / "model.safetensors").stat().st_size == 300_000
    assert not (snap / "model.safetensors.part").exists()


def test_a_server_that_fails_is_retried_then_reported(world, tmp_path, monkeypatch):
    origin, source, _ = world
    monkeypatch.setattr(fs.time, "sleep", lambda s: None)
    del origin.files[f"/{fs.EMBEDDING_REPO}/resolve/{fs.EMBEDDING_REVISION}/config.json"]
    with pytest.raises(fs.FirstStartError, match="could not download config.json"):
        fs.run(tmp_path, source, log=lambda *_: None)
    assert sum("config.json" in h for h in origin.hits) == fs.ATTEMPTS == 8


def test_the_failure_says_to_start_again_and_the_part_is_kept(world, tmp_path, monkeypatch):
    origin, source, served = world
    monkeypatch.setattr(fs.time, "sleep", lambda s: None)
    key = f"/{fs.EMBEDDING_REPO}/resolve/{fs.EMBEDDING_REVISION}/model.safetensors"
    snap = tmp_path / "models--Snowflake--snowflake-arctic-embed-l-v2.0" / "snapshots" / fs.EMBEDDING_REVISION
    snap.mkdir(parents=True)
    (snap / "model.safetensors.part").write_bytes(served[key][:100_000])
    del origin.files[key]
    with pytest.raises(fs.FirstStartError, match="the download continues from where it stopped"):
        fs.run(tmp_path, source, log=lambda *_: None)
    assert (snap / "model.safetensors.part").stat().st_size == 100_000  # nothing was thrown away


def test_a_big_download_reports_how_far_it_is(tmp_path, monkeypatch):
    big = b"B" * 6_000_000
    origin = Origin({"/big": big})
    try:
        monkeypatch.setattr(fs, "PROGRESS_SECONDS", 0)
        lines = []
        fs._fetch(origin.base + "/big", tmp_path / "big.bin", log=lines.append)
    finally:
        origin.close()
    assert (tmp_path / "big.bin").read_bytes() == big
    assert lines[0] == "first-start: big.bin: 6.0 MB"
    progress = [ln for ln in lines[1:] if re.fullmatch(r"first-start: big\.bin: \d\.\d MB of 6\.0 MB \(\d+%\), [\d.]+ MB/s", ln)]
    assert len(progress) >= 2 and progress[-1].endswith("MB/s") and "(100%)" in progress[-1]


def test_a_resumed_download_says_where_it_continues_from(tmp_path, monkeypatch):
    big = b"B" * 6_000_000
    origin = Origin({"/big": big})
    try:
        (tmp_path / "big.bin.part").write_bytes(big[:2_000_000])
        lines = []
        fs._fetch(origin.base + "/big", tmp_path / "big.bin", log=lines.append)
    finally:
        origin.close()
    assert (tmp_path / "big.bin").read_bytes() == big
    assert lines[0] == "first-start: big.bin: 6.0 MB, continuing from 2.0 MB"


def test_a_connection_that_trickles_is_dropped_and_the_next_attempt_resumes(tmp_path, monkeypatch):
    import time as real_time

    body = b"B" * 400_000
    seen = []

    class Trickle(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(re.match(r"bytes=(\d+)-", rng).group(1)) if rng else 0
            seen.append(start)
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            if len(seen) == 1:  # the first connection sends a little, goes quiet, then a little more
                self.wfile.write(body[:100_000])
                self.wfile.flush()
                real_time.sleep(2.5)
                self.wfile.write(body[100_000:100_100])
                self.wfile.flush()
                real_time.sleep(0.5)
            else:
                self.wfile.write(body[start:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(fs, "STALL_SECONDS", 1)
        monkeypatch.setattr(fs, "STALL_MIN_BYTES", 500_000)
        # no waiting between attempts; the clock itself stays real (the server's sleeps use the same module)
        monkeypatch.setattr(fs, "time", SimpleNamespace(monotonic=real_time.monotonic, sleep=lambda s: None))
        lines = []
        fs._fetch(f"http://127.0.0.1:{server.server_port}/f", tmp_path / "f.bin", log=lines.append)
    finally:
        server.shutdown()
    assert (tmp_path / "f.bin").read_bytes() == body
    assert any("attempt 1 of 8 failed (TimeoutError: stalled" in ln for ln in lines)
    assert seen[0] == 0 and seen[1] > 0  # the second attempt asked only for what was missing


def test_a_connection_closed_early_is_a_failed_attempt_not_a_finished_file(tmp_path, monkeypatch):
    body = b"B" * 400_000
    seen = []

    class Cut(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(re.match(r"bytes=(\d+)-", rng).group(1)) if rng else 0
            seen.append(start)
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            self.wfile.write(body[start:start + 150_000] if len(seen) == 1 else body[start:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Cut)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(fs.time, "sleep", lambda s: None)
        lines = []
        fs._fetch(f"http://127.0.0.1:{server.server_port}/f", tmp_path / "f.bin", log=lines.append)
    finally:
        server.shutdown()
    assert (tmp_path / "f.bin").read_bytes() == body
    assert any("attempt 1 of 8 failed" in ln and "closed early" in ln for ln in lines)
    assert seen == [0, 150_000]


def test_a_link_that_keeps_cutting_off_but_keeps_moving_still_finishes(tmp_path, monkeypatch):
    body = b"B" * 400_000
    seen = []

    class CutEveryTime(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(re.match(r"bytes=(\d+)-", rng).group(1)) if rng else 0
            seen.append(start)
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            self.wfile.write(body[start:start + 40_000])  # always short of the end

    server = ThreadingHTTPServer(("127.0.0.1", 0), CutEveryTime)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(fs.time, "sleep", lambda s: None)
        fs._fetch(f"http://127.0.0.1:{server.server_port}/f", tmp_path / "f.bin", log=lambda *_: None)
    finally:
        server.shutdown()
    assert (tmp_path / "f.bin").read_bytes() == body
    assert len(seen) == 10 > fs.ATTEMPTS  # more cut-offs than attempts, each one moved the part forward


def test_attempts_that_get_nowhere_still_run_out(tmp_path, monkeypatch):
    class Dead(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "400000")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Dead)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(fs.time, "sleep", lambda s: None)
        lines = []
        with pytest.raises(fs.FirstStartError, match="could not download"):
            fs._fetch(f"http://127.0.0.1:{server.server_port}/f", tmp_path / "f.bin", log=lines.append)
    finally:
        server.shutdown()
    assert sum("failed" in ln for ln in lines) == fs.ATTEMPTS


def test_a_wheel_with_an_entry_outside_the_folder_is_refused(world, tmp_path, monkeypatch):
    origin, source, served = world
    bad = _wheel({"../escape.txt": "no"})
    served[f"/{fs.SPACY_WHEEL}"] = bad
    monkeypatch.setattr(fs, "SPACY_SHA256", hashlib.sha256(bad).hexdigest())
    with pytest.raises(fs.FirstStartError, match="unsafe path"):
        fs.run(tmp_path, source, log=lambda *_: None)
    assert not (tmp_path.parent / "escape.txt").exists()


def test_the_hub_library_finds_what_was_laid_out(world, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    origin, source, _ = world
    fs.run(tmp_path, source, log=lambda *_: None)
    p = hub.hf_hub_download(fs.EMBEDDING_REPO, "config.json", revision=fs.EMBEDDING_REVISION,
                            cache_dir=str(tmp_path), local_files_only=True)
    assert p.endswith("config.json")


# --- the model settings are checked before anything is downloaded ----------------------------------------

def test_no_model_chosen_is_refused_with_what_to_set(monkeypatch):
    for v in ("SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "SYNAP_LLM_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(fs.FirstStartError, match="no default model: set SYNAP_LLM_PROVIDER and SYNAP_LLM_MODEL"):
        fs.check_llm_settings()


def test_a_placeholder_left_in_the_env_file_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("SYNAP_LLM_PROVIDER", "add-your-provider")
    monkeypatch.setenv("SYNAP_LLM_MODEL", "add-your-model")
    with pytest.raises(fs.FirstStartError, match="SYNAP_LLM_PROVIDER still holds the placeholder"):
        fs.check_llm_settings()


def test_a_chosen_model_passes(monkeypatch):
    monkeypatch.setenv("SYNAP_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("SYNAP_LLM_MODEL", "anything")
    fs.check_llm_settings()


def test_main_stops_before_downloading_when_no_model_is_chosen(monkeypatch, tmp_path, capsys):
    for v in ("SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(fs, "run", lambda *a, **k: pytest.fail("downloaded before checking the model settings"))
    assert fs.main() == 1
    assert "no default model" in capsys.readouterr().err
