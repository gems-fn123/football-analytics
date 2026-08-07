"""Polite, auditable fetching shared by every ingestor.

Rules, non-negotiable by design:
- robots.txt is checked per host for OUR user agent and obeyed. A disallowed
  URL raises rather than fetches; we do not impersonate browsers to get around
  blocks.
- Rate limited per host (default one request per 2 s).
- Every successful response is snapshotted verbatim under
  data/raw/corpus/<source>/<YYYY-MM-DD>/ before any parsing. Re-runs on the
  same day reuse the snapshot instead of re-fetching, so parse iterations are
  free and the site sees each page once.
"""

from __future__ import annotations

import re
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from footy.logging_utils import get_logger

USER_AGENT = "footy-corpus/0.1 (research/stats ingest; contact: ngmacanai@gmail.com)"

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class RobotsDisallowed(RuntimeError):
    pass


class PoliteFetcher:
    def __init__(
        self,
        source: str,
        raw_root: str | Path = "data/raw/corpus",
        min_interval_s: float = 2.0,
        timeout_s: float = 30.0,
        respect_robots: bool = True,
    ) -> None:
        self.source = source
        self.raw_root = Path(raw_root)
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        # robots.txt governs crawling of public pages. A key-authenticated API
        # accessed under its own terms of service is not crawling - FootyStats,
        # for instance, serves "Disallow: /" on the very API it sells access
        # to. Ingestors for licensed APIs set respect_robots=False; scrapers of
        # public pages NEVER do.
        self.respect_robots = respect_robots
        self.log = get_logger(f"footy.corpus.{source}")
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_hit: dict[str, float] = {}

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        # Fetched with OUR user agent, not robotparser.read()'s default
        # Python-urllib (some hosts 403 that, which robotparser then treats as
        # disallow-everything - the opposite of the RFC 9309 unavailable-means-
        # allow rule, and a false negative for hosts that DO publish rules).
        host = urlparse(url).netloc
        if host not in self._robots:
            import requests

            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = requests.get(
                    f"https://{host}/robots.txt",
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout_s,
                )
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:  # no readable rules published -> RFC 9309: allow
                    self.log.warning(
                        "robots.txt for %s returned %d; treating as allow-all",
                        host,
                        resp.status_code,
                    )
                    rp.parse([])
            except Exception as exc:
                self.log.warning("robots.txt unreachable for %s (%s); assuming allow", host, exc)
                rp.parse([])
            self._robots[host] = rp
        return self._robots[host]

    def _snapshot_path(self, key: str) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.raw_root / self.source / day / f"{_SAFE.sub('_', key)}.raw"

    def fetch(self, url: str, key: str) -> str:
        """Fetch url (or reuse today's snapshot). Returns response text."""
        snap = self._snapshot_path(key)
        if snap.exists():
            return snap.read_text(encoding="utf-8")

        if self.respect_robots and not self._robots_for(url).can_fetch(USER_AGENT, url):
            raise RobotsDisallowed(f"{url} disallowed for {USER_AGENT!r}")

        import requests

        host = urlparse(url).netloc
        wait = self.min_interval_s - (time.monotonic() - self._last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=self.timeout_s)
        self._last_hit[host] = time.monotonic()
        resp.raise_for_status()
        # Without a charset in the Content-Type header requests decodes as
        # latin-1 (RFC 2616 default), mojibake-ing names ("Divisi├│n") that
        # entity resolution depends on. Trust the body's own detection then.
        if "charset" not in resp.headers.get("Content-Type", "").lower():
            resp.encoding = resp.apparent_encoding

        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(resp.text, encoding="utf-8")
        logged = re.sub(r"(key=)[^&]+", r"\1***", url)  # never log credentials
        self.log.info("fetched %s -> %s (%d bytes)", logged, snap.name, len(resp.text))
        return resp.text
