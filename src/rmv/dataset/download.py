"""Streaming dataset downloads with checksum verification and resume support."""

from __future__ import annotations

import logging
import re
import tarfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from rmv.dataset.checksums import (
    CSPB_ARCHIVE_EXTENSIONS,
    CSPB_PAGE_URLS,
    CSPB_R2_TRUTH_FILENAME,
    CSPB_R2_TRUTH_URL,
    DEFAULT_DOWNLOAD_TIMEOUT_SEC,
    HISARMOD_GITHUB_RELEASES_API,
    RADIOML_DOWNLOAD_URLS,
    RADIOML_MANUAL_URL,
    RADIOML_ZENODO_RECORD,
    is_verified_checksum,
    sha256_file,
    verify_file_checksum,
)
from rmv.dataset.cspb_detect import (
    CSPBVariant,
    analyze_cspb_directory,
    is_cspb_r2_download_link,
)
from rmv.dataset.manifest import (
    get_expected_checksum,
    update_cspb_manifest,
    update_hisarmod_manifest,
    update_radioml_manifest,
)
from rmv.dataset.paths import (
    RADIOML_PKL_NAME,
    RADIOML_TAR_NAME,
    TRUTH_FILE_NAMES,
    cspb_dir,
    detect_cspb,
    detect_cspb_present,
    find_cspb_truth_file,
    detect_hisarmod,
    detect_radioml,
    hisarmod_h5_path,
    radioml_dir,
    radioml_pkl_path,
    radioml_tar_path,
)
from rmv.dataset.radioml_resolve import RadioMLPickleNotFoundError, extract_radioml_tar_strict

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE_SEC = 2.0
CHUNK_SIZE = 1024 * 256

console = Console(stderr=True)

_download_timeout_sec: float = DEFAULT_DOWNLOAD_TIMEOUT_SEC


class DownloadError(Exception):
    """User-facing download failure (no traceback in CLI)."""


def set_download_timeout(seconds: float) -> None:
    """Configure HTTP timeout for subsequent downloads in this process."""
    global _download_timeout_sec
    _download_timeout_sec = max(30.0, seconds)


def get_download_timeout() -> httpx.Timeout:
    """Build httpx.Timeout for large dataset transfers."""
    return httpx.Timeout(_download_timeout_sec, connect=30.0)


def _retry_delay(attempt: int) -> float:
    return BACKOFF_BASE_SEC * (2**attempt)


def _remote_content_length(client: httpx.Client, url: str) -> int | None:
    """HEAD request to learn total object size for resume completeness checks."""
    try:
        response = client.head(url)
        if response.status_code >= 400:
            return None
        cl = response.headers.get("Content-Length")
        return int(cl) if cl else None
    except httpx.HTTPError:
        return None


def is_download_complete(
    dest: Path,
    expected_sha256: str | None,
    expected_size: int | None,
) -> bool:
    """True only when file is fully present (checksum or size match), not partial."""
    if not dest.is_file():
        return False
    size = dest.stat().st_size
    if expected_sha256 and is_verified_checksum(expected_sha256):
        return sha256_file(dest).lower() == expected_sha256.lower()
    if expected_size is not None and expected_size > 0:
        return size >= expected_size
    return False


def stream_download(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
    resume: bool = True,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
    datasets_root: Path | None = None,
) -> Path:
    """
    Stream URL to dest with optional resume, retries, and SHA-256 verification.

    Partial files are resumed via Range; never treated as complete without verification.
    Deletes dest on checksum failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True, timeout=get_download_timeout()) as client:
        remote_size = _remote_content_length(client, url)

    if dest.is_file() and not force:
        if is_download_complete(dest, expected_sha256, remote_size):
            return dest

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            _download_once(
                url,
                dest,
                force=force,
                resume=resume,
                progress=progress,
                task_id=task_id,
                remote_size=remote_size,
            )
            if expected_sha256 and is_verified_checksum(expected_sha256):
                actual = sha256_file(dest)
                if actual.lower() != expected_sha256.lower():
                    dest.unlink(missing_ok=True)
                    msg = (
                        f"Checksum mismatch for {dest.name}: "
                        f"expected {expected_sha256[:16]}..., got {actual[:16]}..."
                    )
                    raise DownloadError(msg)
            elif remote_size and dest.stat().st_size < remote_size:
                msg = (
                    f"Incomplete download for {dest.name}: "
                    f"{dest.stat().st_size} of {remote_size} bytes"
                )
                raise DownloadError(msg)
            return dest
        except (httpx.HTTPError, OSError, DownloadError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                delay = _retry_delay(attempt)
                logger.warning("Download attempt %d failed: %s; retry in %.1fs", attempt + 1, exc, delay)
                time.sleep(delay)
            else:
                break

    msg = f"Failed to download {url} after {MAX_RETRIES} attempts: {last_error}"
    raise DownloadError(msg) from last_error


def _download_once(
    url: str,
    dest: Path,
    *,
    force: bool,
    resume: bool,
    progress: Progress | None,
    task_id: TaskID | None,
    remote_size: int | None,
) -> None:
    start_pos = 0
    mode = "wb"
    headers: dict[str, str] = {}

    if dest.is_file() and resume and not force:
        start_pos = dest.stat().st_size
        if remote_size and start_pos >= remote_size:
            return
        if start_pos > 0:
            headers["Range"] = f"bytes={start_pos}-"
            mode = "ab"

    with httpx.Client(follow_redirects=True, timeout=get_download_timeout()) as client:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416:
                if remote_size and dest.is_file() and dest.stat().st_size >= remote_size:
                    return
                raise DownloadError(f"Range not satisfiable for {dest.name}")

            if start_pos > 0 and response.status_code == 200:
                dest.unlink(missing_ok=True)
                start_pos = 0
                mode = "wb"
                headers = {}
                with client.stream("GET", url) as response2:
                    response2.raise_for_status()
                    _write_stream(response2, dest, mode, 0, progress, task_id, remote_size)
                return

            if start_pos > 0 and response.status_code not in (200, 206):
                dest.unlink(missing_ok=True)
                with client.stream("GET", url) as response2:
                    response2.raise_for_status()
                    _write_stream(response2, dest, "wb", 0, progress, task_id, remote_size)
                return

            response.raise_for_status()
            _write_stream(response, dest, mode, start_pos, progress, task_id, remote_size)


def _write_stream(
    response: httpx.Response,
    dest: Path,
    mode: str,
    start_pos: int,
    progress: Progress | None,
    task_id: TaskID | None,
    remote_size: int | None,
) -> None:
    total: int | None = remote_size
    content_length = response.headers.get("Content-Length")
    if content_length:
        chunk_len = int(content_length)
        total = chunk_len + start_pos if response.status_code == 206 else chunk_len
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        try:
            total = int(content_range.split("/")[-1])
        except ValueError:
            pass

    if progress is not None and task_id is not None:
        progress.update(task_id, total=total, completed=start_pos)

    downloaded = start_pos
    with dest.open(mode) as f:
        for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if progress is not None and task_id is not None:
                    progress.update(task_id, completed=downloaded)

    if progress is not None and task_id is not None:
        progress.update(task_id, completed=downloaded)


def extract_radioml_tar(tar_path: Path, dest_dir: Path) -> Path:
    """Extract RadioML tar.bz2; return RML2016.10a_dict.pkl (strict)."""
    return extract_radioml_tar_strict(tar_path, dest_dir)


def _download_radioml_archive(
    tar_path: Path,
    *,
    expected_sha256: str | None,
    force: bool,
    progress: Progress | None,
    datasets_root: Path,
) -> bool:
    """Try each known RadioML mirror until one succeeds."""
    last_error: Exception | None = None
    for i, url in enumerate(RADIOML_DOWNLOAD_URLS):
        label = f"{RADIOML_TAR_NAME} ({i + 1}/{len(RADIOML_DOWNLOAD_URLS)})"
        task_id = progress.add_task(label, total=None) if progress else None
        try:
            console.print(f"[cyan]Trying RadioML mirror:[/] {url}")
            stream_download(
                url,
                tar_path,
                expected_sha256=expected_sha256,
                force=force,
                progress=progress,
                task_id=task_id,
                datasets_root=datasets_root,
            )
            return True
        except DownloadError as exc:
            last_error = exc
            console.print(f"[yellow]Mirror failed:[/] {exc}")
            tar_path.unlink(missing_ok=True)
    console.print(
        "[yellow]All automatic RadioML mirrors failed.[/]\n"
        f"Manual options:\n"
        f"  1. {RADIOML_MANUAL_URL}\n"
        f"  2. Zenodo mirror: {RADIOML_ZENODO_RECORD}\n"
        f"Place RML2016.10a.tar.bz2 at: {tar_path}\n"
        "Then run: [bold]rmv dataset verify[/]"
    )
    if last_error:
        logger.debug("Last RadioML error: %s", last_error)
    return tar_path.is_file()


def download_radioml(
    root: Path,
    *,
    force: bool = False,
    verify_only: bool = False,
    progress: Progress | None = None,
) -> bool:
    """Download and extract RadioML 2016.10A. Returns True on success."""
    rdir = radioml_dir(root)
    tar_path = radioml_tar_path(root)
    pkl_path = radioml_pkl_path(root)
    tar_key = "radioml/RML2016.10a.tar.bz2"
    pkl_key = "radioml/RML2016.10a_dict.pkl"
    expected_tar = get_expected_checksum(root, tar_key)
    expected_pkl = get_expected_checksum(root, pkl_key)

    if detect_radioml(root) and not force and not verify_only:
        console.print(f"[green]RadioML already present:[/] {detect_radioml(root)}")
        return True

    if verify_only:
        if not tar_path.is_file() and not pkl_path.is_file():
            return False
        for path, key in ((tar_path, tar_key), (pkl_path, pkl_key)):
            if path.is_file():
                ok, msg = verify_file_checksum(path, key, datasets_root=root)
                if not ok:
                    raise DownloadError(f"RadioML verification failed: {path.name} ({msg})")
        return True

    if not tar_path.is_file() or force:
        if not _download_radioml_archive(
            tar_path,
            expected_sha256=expected_tar,
            force=force,
            progress=progress,
            datasets_root=root,
        ):
            return False
    else:
        console.print(f"[green]Using existing[/] {tar_path}")

    if not pkl_path.is_file():
        console.print("Extracting RadioML archive...")
        try:
            extract_radioml_tar(tar_path, rdir)
        except RadioMLPickleNotFoundError as exc:
            raise DownloadError(str(exc)) from exc

    tar_sha = sha256_file(tar_path) if tar_path.is_file() else None
    pkl_sha = sha256_file(pkl_path) if pkl_path.is_file() else None
    if expected_pkl and pkl_path.is_file() and is_verified_checksum(expected_pkl):
        if pkl_sha and pkl_sha.lower() != expected_pkl.lower():
            pkl_path.unlink(missing_ok=True)
            raise DownloadError("RadioML pickle checksum mismatch after extraction")

    update_radioml_manifest(root, tar_sha256=tar_sha, pkl_sha256=pkl_sha, status="verified")
    console.print(f"[green]RadioML ready:[/] {detect_radioml(root)}")
    return True


def _find_hisarmod_github_asset() -> tuple[str, str] | None:
    with httpx.Client(timeout=get_download_timeout()) as client:
        response = client.get(HISARMOD_GITHUB_RELEASES_API)
        response.raise_for_status()
        releases: list[dict[str, Any]] = response.json()

    pattern = re.compile(r"hisarmod.*\.h5$|HisarMod.*\.h5$", re.IGNORECASE)
    for release in releases:
        for asset in release.get("assets", []):
            name = str(asset.get("name", ""))
            if pattern.search(name):
                url = str(asset.get("browser_download_url", ""))
                if url:
                    return url, name
    return None


HISARMOD_MANUAL_INSTRUCTIONS = """
HISARMOD requires IEEE DataPort access.
  1. Go to https://ieee-dataport.org/open-access/hisarmod-new-challenging-modulated-signals-dataset
  2. Log in with an IEEE account (free for IEEE members)
  3. Download HisarMod2019.1.h5 or the .mat files
  4. Place the file at: {path}
  Then run: rmv dataset verify
"""


def download_hisarmod(
    root: Path,
    *,
    force: bool = False,
    verify_only: bool = False,
    progress: Progress | None = None,
) -> bool:
    h5_path = hisarmod_h5_path(root)
    h5_key = "hisarmod/HisarMod2019.1.h5"
    expected = get_expected_checksum(root, h5_key)

    if detect_hisarmod(root) and not force and not verify_only:
        console.print(f"[green]HISARMOD already present:[/] {detect_hisarmod(root)}")
        return True

    if verify_only:
        if not h5_path.is_file():
            return False
        ok, msg = verify_file_checksum(h5_path, h5_key, datasets_root=root)
        if not ok:
            raise DownloadError(f"HISARMOD verification failed ({msg})")
        return True

    if h5_path.is_file() and not force:
        console.print(f"[green]Using existing[/] {h5_path}")
        return True

    asset = _find_hisarmod_github_asset()
    if asset is None:
        console.print(HISARMOD_MANUAL_INSTRUCTIONS.format(path=h5_path))
        return h5_path.is_file()

    url, name = asset
    console.print(f"[cyan]Found IQFormer mirror asset:[/] {name}")
    task_id = progress.add_task(name, total=None) if progress else None
    try:
        stream_download(
            url,
            h5_path,
            expected_sha256=expected,
            force=force,
            progress=progress,
            task_id=task_id,
            datasets_root=root,
        )
    except DownloadError:
        console.print(HISARMOD_MANUAL_INSTRUCTIONS.format(path=h5_path))
        return False

    update_hisarmod_manifest(root, sha256_file(h5_path), status="verified")
    console.print(f"[green]HISARMOD ready:[/] {h5_path}")
    return True


CSPB_MANUAL_INSTRUCTIONS = """
CSPB.ML.2018R2 batch files must be downloaded manually.
  1. Go to https://cyclostationary.blog/data-sets/
  2. Find and download all CSPB.ML.2018R2 batch files (not original 2018)
  3. Place them in: {path}
  4. Download the R2 truth file (signal_record_C_2023.txt) from the R2 post
     (link text: "The new metadata ... can be found here") into {path}/
     Run: rmv dataset download --cspb  (fetches truth automatically when possible)
  Then run: rmv dataset verify
"""

CSPB_RNG_WARNING = """
WARNING: Original CSPB.ML.2018 has a known RNG flaw.
Use CSPB.ML.2018R2 from the correction post.
See: https://cyclostationary.blog/2023/09/25/cspb-ml-2018r2-correcting-an-rng-flaw-in-cspb-ml-2018/
"""


def _scrape_cspb_download_links() -> list[tuple[str, str]]:
    """Parse CSP blog pages for R2-only download links (reject original 2018)."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    with httpx.Client(follow_redirects=True, timeout=get_download_timeout()) as client:
        for page_url in CSPB_PAGE_URLS:
            try:
                response = client.get(page_url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Could not fetch %s: %s", page_url, exc)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup.find_all("a", href=True):
                href = str(tag["href"])
                text = tag.get_text(strip=True)
                full_url = urljoin(page_url, href)
                lower = (href + " " + text).lower()
                if not any(ext in lower for ext in CSPB_ARCHIVE_EXTENSIONS + (".zip", ".gz", ".bz2")):
                    continue
                if not is_cspb_r2_download_link(full_url, text):
                    logger.debug("Skipping non-R2 CSPB link: %s", full_url)
                    continue
                if full_url in seen:
                    continue
                seen.add(full_url)
                name = Path(urlparse(full_url).path).name or "cspb_r2_batch.zip"
                links.append((full_url, name))

    return links


def _scrape_cspb_truth_links() -> list[tuple[str, str]]:
    """Find truth/metadata .txt links on CSPB blog pages (R2 file preferred)."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    with httpx.Client(follow_redirects=True, timeout=get_download_timeout()) as client:
        for page_url in CSPB_PAGE_URLS:
            try:
                response = client.get(page_url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Could not fetch %s: %s", page_url, exc)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup.find_all("a", href=True):
                href = str(tag["href"])
                full_url = urljoin(page_url, href)
                lower = (href + " " + tag.get_text(strip=True)).lower()
                path_lower = urlparse(full_url).path.lower()
                if not path_lower.endswith(".txt"):
                    continue
                if "signal_record" not in lower and "metadata" not in lower:
                    continue
                if full_url in seen:
                    continue
                seen.add(full_url)
                name = Path(urlparse(full_url).path).name or "signal_record.txt"
                links.append((full_url, name))

    if not links:
        links.append((CSPB_R2_TRUTH_URL, CSPB_R2_TRUTH_FILENAME))

    def _truth_priority(item: tuple[str, str]) -> int:
        url, name = item
        combined = (url + name).lower()
        if "c_2023" in combined or "2018r2" in combined:
            return 0
        if "2023/09" in combined:
            return 1
        if "signal_record.txt" == name:
            return 2
        return 3

    links.sort(key=_truth_priority)
    return links


def download_cspb_truth_file(
    root: Path,
    *,
    force: bool = False,
    progress: Progress | None = None,
) -> Path | None:
    """Download CSPB.ML.2018R2 truth/metadata file into datasets/cspb/."""
    cdir = cspb_dir(root)
    cdir.mkdir(parents=True, exist_ok=True)

    existing = find_cspb_truth_file(cdir)
    if existing is not None and not force:
        console.print(f"[green]CSPB truth file already present:[/] {existing}")
        return existing

    canonical = cdir / "signal_record.txt"
    for url, name in _scrape_cspb_truth_links():
        dest = cdir / name
        task_id = progress.add_task(name, total=None) if progress else None
        try:
            stream_download(url, dest, force=force, progress=progress, task_id=task_id)
        except DownloadError as exc:
            console.print(f"[yellow]Skipped truth file {name}:[/] {exc}")
            continue
        if dest.stat().st_size < 20:
            dest.unlink(missing_ok=True)
            continue
        if name != canonical.name and not canonical.is_file():
            canonical.write_bytes(dest.read_bytes())
        console.print(f"[green]CSPB truth file downloaded:[/] {dest}")
        return dest

    console.print(
        "[yellow]Could not download CSPB truth file.[/]\n"
        f"Manual: open {CSPB_PAGE_URLS[0]}\n"
        f"Download the metadata link, save as {cdir}/signal_record_C_2023.txt"
    )
    return None


def _extract_cspb_archive(archive: Path, dest: Path) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest, filter="data")
    elif name.endswith((".tar.bz2", ".tbz2")):
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(dest, filter="data")
    elif name.endswith(".tar"):
        with tarfile.open(archive, "r:") as tar:
            tar.extractall(dest, filter="data")


def _ensure_cspb_truth_file(cdir: Path) -> bool:
    return find_cspb_truth_file(cdir) is not None


def download_cspb(
    root: Path,
    *,
    force: bool = False,
    verify_only: bool = False,
    progress: Progress | None = None,
) -> bool:
    cdir = cspb_dir(root)
    cdir.mkdir(parents=True, exist_ok=True)

    variant = analyze_cspb_directory(cdir)
    if variant == CSPBVariant.ORIGINAL:
        console.print(f"[bold red]{CSPB_RNG_WARNING.strip()}[/]")

    if detect_cspb(root) and not force and not verify_only:
        if variant != CSPBVariant.ORIGINAL:
            console.print(f"[green]CSPB.ML.2018R2 already present:[/] {cdir}")
            return True
    if detect_cspb_present(root) and not force and not verify_only:
        if variant != CSPBVariant.ORIGINAL:
            if find_cspb_truth_file(cdir) is None:
                download_cspb_truth_file(root, force=force, progress=progress)
            if detect_cspb(root):
                console.print(f"[green]CSPB.ML.2018R2 ready:[/] {cdir}")
                return True
            console.print(
                f"[yellow]CSPB.ML.2018R2 signals present at {cdir}; "
                "truth file still missing (see R2 post metadata link)"
            )
            return True

    if verify_only:
        if not _ensure_cspb_truth_file(cdir):
            return False
        if variant == CSPBVariant.ORIGINAL:
            raise DownloadError("CSPB original 2018 detected; use R2 dataset only")
        return True

    links = _scrape_cspb_download_links()
    if not links:
        console.print(CSPB_MANUAL_INSTRUCTIONS.format(path=cdir))
        return detect_cspb(root) is not None

    for url, name in links:
        if not is_cspb_r2_download_link(url, name):
            continue
        dest = cdir / name
        if dest.is_file() and not force:
            continue
        task_id = progress.add_task(name, total=None) if progress else None
        try:
            stream_download(url, dest, force=force, progress=progress, task_id=task_id)
            if dest.suffix in (".zip", ".gz", ".bz2", ".tar") or ".tar." in dest.name:
                _extract_cspb_archive(dest, cdir)
        except DownloadError as exc:
            console.print(f"[yellow]Skipped {name}:[/] {exc}")

    if analyze_cspb_directory(cdir) == CSPBVariant.ORIGINAL:
        console.print(f"[bold red]{CSPB_RNG_WARNING.strip()}[/]")
        raise DownloadError("Only original CSPB.ML.2018 files found; R2 required")

    if not _ensure_cspb_truth_file(cdir):
        download_cspb_truth_file(root, force=force, progress=progress)

    if detect_cspb(root):
        n_tim = len(list(cdir.glob("signal_*.tim"))) + len(list(cdir.rglob("*.tim")))
        update_cspb_manifest(root, batch_files=n_tim, status="verified", version="R2")
        console.print(f"[green]CSPB ready:[/] {cdir}")
        return True

    console.print(CSPB_MANUAL_INSTRUCTIONS.format(path=cdir))
    return False


def download_datasets(
    root: Path,
    *,
    radioml: bool = True,
    hisarmod: bool = True,
    cspb: bool = True,
    force: bool = False,
    verify_only: bool = False,
) -> bool:
    steps: list[tuple[str, Callable[..., bool]]] = []
    if radioml:
        steps.append(
            ("RadioML 2016.10A", lambda p: download_radioml(root, force=force, verify_only=verify_only, progress=p))
        )
    if hisarmod:
        steps.append(
            ("HISARMOD 2019.1", lambda p: download_hisarmod(root, force=force, verify_only=verify_only, progress=p))
        )
    if cspb:
        steps.append(
            ("CSPB.ML.2018R2", lambda p: download_cspb(root, force=force, verify_only=verify_only, progress=p))
        )

    all_ok = True
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        for i, (name, fn) in enumerate(steps, start=1):
            console.print(f"\n[bold]Downloading datasets (step {i}/{len(steps)}: {name})[/]")
            try:
                ok = fn(progress)
                all_ok = all_ok and ok
            except DownloadError as exc:
                console.print(f"[red]Error:[/] {exc}")
                all_ok = False

    return all_ok


def is_matlab_v73(path: Path) -> bool:
    """Detect MATLAB v7.3 (HDF5-based) .mat files."""
    with path.open("rb") as f:
        header = f.read(8)
    return header == b"\x89HDF\r\n\x1a\n"


def _convert_mat_v73_to_h5(src: Path, dst: Path) -> Path:
    """Copy HDF5 structure from MATLAB v7.3 .mat to standalone .h5."""
    import h5py

    dst.parent.mkdir(parents=True, exist_ok=True)

    def copy_item(name: str, obj: object, out_parent: h5py.Group) -> None:
        import h5py as h5

        if isinstance(obj, h5.Dataset):
            out_parent.create_dataset(name, data=obj[()])
        elif isinstance(obj, h5.Group):
            grp = out_parent.create_group(name)
            for key in obj.keys():
                copy_item(key, obj[key], grp)

    with h5py.File(src, "r") as src_f, h5py.File(dst, "w") as dst_f:
        for key in src_f.keys():
            copy_item(key, src_f[key], dst_f)

    return dst


def convert_hisarmod_mat_to_h5(input_path: Path, output_path: Path) -> Path:
    """
    Convert HISARMOD .mat to HDF5.

    HisarMod2019.1 uses MATLAB v7.3 (HDF5); scipy.io.loadmat cannot read it.
    """
    if not input_path.is_file():
        raise DownloadError(f"Input file not found: {input_path}")

    console.print(f"Converting {input_path} -> {output_path}")

    if is_matlab_v73(input_path):
        console.print("Detected MATLAB v7.3 (HDF5) format; using h5py")
        return _convert_mat_v73_to_h5(input_path, output_path)

    import scipy.io

    try:
        mat = scipy.io.loadmat(str(input_path))
    except NotImplementedError as exc:
        msg = (
            "scipy.io.loadmat cannot read this file (likely MATLAB v7.3). "
            "Use the IEEE HDF5 download or: rmv dataset convert-hisarmod --input FILE.h5"
        )
        raise DownloadError(msg) from exc

    import h5py

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            try:
                hf.create_dataset(key, data=value)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping key %s: %s", key, exc)

    console.print(f"[green]Wrote[/] {output_path}")
    return output_path
