from __future__ import annotations

import email.utils
import json
import time
import urllib.robotparser
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.parse import urlparse


class NetworkError(RuntimeError):
    pass


class RobotsDenied(NetworkError):
    pass


def require_requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Dataset collection requires the optional dependencies. "
            "Install with `python -m pip install -e '.[dataset]'`."
        ) from exc
    return requests


@dataclass(slots=True)
class HTTPResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes
    url: str

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


class PoliteHTTPClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 30.0,
        requests_per_second: float = 1.0,
        retries: int = 3,
        backoff_seconds: float = 1.0,
    ):
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.requests_per_second = max(0.01, requests_per_second)
        self.retries = max(0, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self._last_request_by_host: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = Lock()

    def _rate_limit(self, url: str) -> None:
        host = (urlparse(url).hostname or "").lower()
        with self._lock:
            last = self._last_request_by_host.get(host, 0.0)
            wait = (1.0 / self.requests_per_second) - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_request_by_host[host] = time.monotonic()

    def allowed_by_robots(self, url: str, *, fail_closed: bool = False) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            parser = urllib.robotparser.RobotFileParser(f"{origin}/robots.txt")
            try:
                requests = require_requests()
                self._rate_limit(parser.url)
                response = requests.get(
                    parser.url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                parser.parse(response.text.splitlines())
                self._robots[origin] = parser
            except Exception:
                self._robots[origin] = None
        robots = self._robots[origin]
        if robots is None:
            return not fail_closed
        return bool(robots.can_fetch(self.user_agent, url))

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        check_robots: bool = False,
        robots_fail_closed: bool = False,
    ) -> HTTPResponse:
        requests = require_requests()
        if check_robots and not self.allowed_by_robots(url, fail_closed=robots_fail_closed):
            raise RobotsDenied(f"robots.txt does not permit collection of {url}")
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json, image/*, video/*;q=0.9, */*;q=0.5"}
        request_headers.update(headers or {})
        last_error = "unknown network failure"
        for attempt in range(self.retries + 1):
            self._rate_limit(url)
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=request_headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    self._wait_for_retry(response.headers.get("Retry-After"), attempt)
                    continue
                response.raise_for_status()
                return HTTPResponse(
                    response.status_code,
                    {str(k): str(v) for k, v in response.headers.items()},
                    response.content,
                    str(response.url),
                )
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                last_error = exc.__class__.__name__ + (f" (HTTP {status})" if status else "")
                if attempt >= self.retries:
                    break
                self._wait_for_retry(None, attempt)
        raise NetworkError(f"GET failed after {self.retries + 1} attempts for {url}: {last_error}")

    def _wait_for_retry(self, retry_after: str | None, attempt: int) -> None:
        delay = self.backoff_seconds * (2**attempt)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                try:
                    parsed = email.utils.parsedate_to_datetime(retry_after)
                    delay = max(delay, parsed.timestamp() - time.time())
                except (TypeError, ValueError, OverflowError):
                    pass
        time.sleep(max(0.0, min(delay, 60.0)))
