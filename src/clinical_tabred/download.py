"""Download NHANES XPT files from the CDC website, validate them and cache them locally.

The CDC site sometimes answers with an HTML error page (HTTP 200) instead of the
data file, so every file is checked before it is used.
"""

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger("clinical_tabred")

# Every SAS transport (XPT) file starts with this fixed header text.
XPT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD"


def build_urls(download_cfg: dict, cycle: dict, component: str) -> list[str]:
    """Return the candidate download URLs (primary first, then legacy fallback)."""
    fields = {**cycle, "component": component}
    return [download_cfg["primary_url"].format(**fields), download_cfg["fallback_url"].format(**fields)]


def validate_xpt(path: Path) -> str:
    """Check that a file is a real XPT file. Returns "ok" or a short error message."""
    if not path.exists() or path.stat().st_size == 0:
        return "missing or empty file"
    head = path.read_bytes()[:80]
    if head.lstrip().startswith(b"<"):
        return "file is an HTML page, not XPT"
    if not head.startswith(XPT_MAGIC):
        return "missing SAS transport header"
    try:
        pd.read_sas(path, format="xport").head()
    except Exception as error:  # noqa: BLE001 - any parser error means "invalid file"
        return f"pandas cannot read file: {error}"
    return "ok"


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (recorded in the manifest)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_url(url: str, target: Path, download_cfg: dict) -> int:
    """Download one URL to `target` with retries. Returns the final HTTP status code."""
    headers = {"User-Agent": download_cfg["user_agent"]}
    status = -1
    for attempt in range(1, download_cfg["retries"] + 1):
        try:
            response = requests.get(url, headers=headers, timeout=download_cfg["timeout_s"])
            status = response.status_code
            if status == 200:
                target.write_bytes(response.content)
                return status
            logger.warning("HTTP %s for %s (attempt %d)", status, url, attempt)
        except requests.RequestException as error:
            logger.warning("Request failed for %s (attempt %d): %s", url, attempt, error)
        time.sleep(2 * attempt)
    return status


def download_component(
    cycle: dict, component: str, download_cfg: dict, raw_dir: Path, previous: dict
) -> dict:
    """Make sure one valid XPT file is cached in `raw_dir`; return its manifest row.

    `previous` maps file names to rows of an earlier manifest, so a cached file
    keeps the URL and HTTP status from the run that actually downloaded it.
    """
    target = raw_dir / f"{component}_{cycle['suffix']}.xpt"
    row = {"component": component, "cycle": cycle["name"], "file": target.name}
    if validate_xpt(target) == "ok":
        logger.info("Using cached %s", target.name)
        old = previous.get(target.name, {})
        return {**row, "url": old.get("url", "cached (source unknown)"),
                "http_status": old.get("http_status"), "bytes": target.stat().st_size,
                "sha256": sha256_of(target), "validation": "ok", "from_cache": True}
    for url in build_urls(download_cfg, cycle, component):
        status = fetch_url(url, target, download_cfg)
        result = validate_xpt(target) if status == 200 else f"HTTP {status}"
        logger.info("%s -> %s (%s)", url, status, result)
        if result == "ok":
            return {**row, "url": url, "http_status": status, "bytes": target.stat().st_size,
                    "sha256": sha256_of(target), "validation": "ok", "from_cache": False}
        target.unlink(missing_ok=True)  # never keep an invalid file in the cache
    raise RuntimeError(
        f"Could not download a valid {component}_{cycle['suffix']} file. Tried: "
        f"{build_urls(download_cfg, cycle, component)}; last result: {result}. "
        "Check your internet connection or download the file manually into "
        f"{raw_dir}."
    )


def download_all(config: dict, raw_dir: Path) -> pd.DataFrame:
    """Download every component of every cycle and write `manifest.csv` to `raw_dir`."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.csv"
    previous = {}
    if manifest_path.exists():
        previous = pd.read_csv(manifest_path).set_index("file").to_dict("index")
    rows = []
    for cycle in config["cycles"]:
        for component in config["download"]["components"]:
            rows.append(
                download_component(cycle, component, config["download"], raw_dir, previous)
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest
