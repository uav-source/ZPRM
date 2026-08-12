"""Fail-closed guard against Boreas Stage-2 LiDAR payload downloads.

The storage-planning task may inspect S3 metadata but is not authorized to
materialize any ``Boreas/.../lidar/*.bin`` object.  This context manager patches
the ordinary subprocess and HTTP entrypoints used by repository preparation
code while it is active.  It complements, rather than replaces, the existing
``NoRegistrationGuard``.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import urllib.request
from types import TracebackType
from typing import Any, Callable, Sequence
from urllib.parse import unquote, urlsplit


NO_REGISTRATION_ENV = "ZPRM_REAL_DATA_PREP_NO_REGISTRATION"
NO_LIDAR_DOWNLOAD_ENV = "ZPRM_BOREAS_NO_LIDAR_PAYLOAD_DOWNLOAD"
AWS_METADATA_OPERATIONS = frozenset({"ls", "head-object", "list-objects", "list-objects-v2"})
AWS_PAYLOAD_OPERATIONS = frozenset({"get-object", "cp", "sync"})


class LidarPayloadDownloadForbiddenError(PermissionError):
    """A command or HTTP request attempted to materialize Boreas LiDAR bytes."""


def _flatten_command(command: Any) -> tuple[list[str], str]:
    if isinstance(command, (list, tuple)):
        tokens = [os.fspath(item) for item in command]
        return tokens, " ".join(tokens)
    text = os.fspath(command)
    try:
        return shlex.split(text), text
    except ValueError:
        # An unparseable shell command is still inspected conservatively as text.
        return [text], text


def _normalized_command_tokens(command: Any) -> tuple[list[str], str]:
    tokens, text = _flatten_command(command)
    return [token.lower() for token in tokens], text.lower()


def _mentions_boreas_lidar(value: str, *, require_bin: bool) -> bool:
    normalized = unquote(value).lower().replace("\\", "/")
    has_boreas = (
        "s3://boreas/" in normalized
        or "--bucket boreas" in normalized
        or "boreas.s3" in normalized
        or re.search(r"s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com/boreas/", normalized)
        is not None
        or re.search(r"(?:^|[/\s])boreas-[0-9].*?/lidar(?:/|\s|$)", normalized) is not None
    )
    has_lidar = re.search(r"(?:^|[/\s])lidar(?:/|\s|$)", normalized) is not None
    has_bin = re.search(r"(?:^|[/])[^/?\s]+\.bin(?:[?\s]|$)", normalized) is not None
    return bool(has_boreas and has_lidar and (has_bin or not require_bin))


def _aws_operation(tokens: Sequence[str], text: str) -> str | None:
    # The token scan handles normal argv; the regex also catches ``sh -c``.
    for operation in sorted(AWS_METADATA_OPERATIONS | AWS_PAYLOAD_OPERATIONS):
        if operation in tokens:
            return operation
    match = re.search(
        r"(?:^|[;&|\s])aws(?:\s+--?[^\s]+(?:\s+[^\s]+)?)*\s+"
        r"(?:s3|s3api)\s+(get-object|head-object|list-objects-v2|list-objects|cp|sync|ls)\b",
        text,
    )
    return match.group(1) if match else None


def command_attempts_boreas_lidar_download(command: Any) -> bool:
    """Classify AWS/curl/wget commands without executing them."""

    original_tokens, _original_text = _flatten_command(command)
    tokens, text = _normalized_command_tokens(command)
    operation = _aws_operation(tokens, text)
    if operation in AWS_PAYLOAD_OPERATIONS:
        # cp/sync on a lidar prefix is blocked even when no individual .bin key
        # appears in argv; get-object is treated the same way fail closed.
        return _mentions_boreas_lidar(text, require_bin=False)
    if operation in AWS_METADATA_OPERATIONS:
        return False
    executables = {token.rsplit("/", 1)[-1] for token in tokens}
    head_only = any(token in {"-I", "--head"} for token in original_tokens) or any(
        original_tokens[index : index + 2] in (["-X", "HEAD"], ["--request", "HEAD"])
        for index in range(max(0, len(original_tokens) - 1))
    )
    if "curl" in executables:
        return not head_only and _mentions_boreas_lidar(text, require_bin=True)
    if "wget" in executables:
        return "--spider" not in tokens and _mentions_boreas_lidar(text, require_bin=True)
    # A URL handed to another command is also a materialization attempt unless
    # it is clearly an HTTP HEAD invocation.
    if "http://" in text or "https://" in text:
        return not head_only and _mentions_boreas_lidar(text, require_bin=True)
    return False


def url_is_boreas_lidar_payload(url: Any) -> bool:
    """Return true only for an HTTP(S) Boreas ``lidar/*.bin`` object URL."""

    if isinstance(url, urllib.request.Request):
        raw = url.full_url
    else:
        raw = str(url)
    split = urlsplit(raw)
    if split.scheme.lower() not in {"http", "https"}:
        return False
    combined = f"{split.netloc}{unquote(split.path)}"
    return _mentions_boreas_lidar(combined, require_bin=True)


def _http_method(value: Any, default: str = "GET") -> str:
    if isinstance(value, urllib.request.Request):
        return value.get_method().upper()
    return default.upper()


class BoreasLidarPayloadGuard:
    """Patch process and HTTP paths while metadata-only planning is active."""

    def __init__(self) -> None:
        self.active = False
        self.activation_count = 0
        self.blocked_process_attempt_count = 0
        self.blocked_http_attempt_count = 0
        self.allowed_metadata_operation_count = 0
        self._original_subprocess: dict[str, Callable[..., Any]] = {}
        self._original_urlopen: Callable[..., Any] | None = None
        self._original_urlretrieve: Callable[..., Any] | None = None
        self._requests_session_class: Any = None
        self._original_requests_request: Callable[..., Any] | None = None

    def _guard_process(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(command: Any, *args: Any, **kwargs: Any) -> Any:
            tokens, text = _normalized_command_tokens(command)
            operation = _aws_operation(tokens, text)
            if command_attempts_boreas_lidar_download(command):
                self.blocked_process_attempt_count += 1
                raise LidarPayloadDownloadForbiddenError(
                    f"Boreas lidar payload download is forbidden: {text}"
                )
            if operation in AWS_METADATA_OPERATIONS:
                self.allowed_metadata_operation_count += 1
            return original(command, *args, **kwargs)

        return guarded

    def _guard_urlopen(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(url: Any, *args: Any, **kwargs: Any) -> Any:
            method = _http_method(url)
            if method != "HEAD" and url_is_boreas_lidar_payload(url):
                self.blocked_http_attempt_count += 1
                raise LidarPayloadDownloadForbiddenError(
                    f"Boreas lidar HTTP payload download is forbidden: {url}"
                )
            if method == "HEAD":
                self.allowed_metadata_operation_count += 1
            return original(url, *args, **kwargs)

        return guarded

    def _guard_urlretrieve(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(url: Any, *args: Any, **kwargs: Any) -> Any:
            if url_is_boreas_lidar_payload(url):
                self.blocked_http_attempt_count += 1
                raise LidarPayloadDownloadForbiddenError(
                    f"Boreas lidar HTTP payload download is forbidden: {url}"
                )
            return original(url, *args, **kwargs)

        return guarded

    def _guard_requests(self) -> None:
        try:
            import requests.sessions  # type: ignore[import-not-found]
        except ImportError:
            return
        session_class = requests.sessions.Session
        original = session_class.request

        def guarded_request(
            session: Any, method: str, url: Any, *args: Any, **kwargs: Any
        ) -> Any:
            normalized_method = str(method).upper()
            if normalized_method != "HEAD" and url_is_boreas_lidar_payload(url):
                self.blocked_http_attempt_count += 1
                raise LidarPayloadDownloadForbiddenError(
                    f"Boreas lidar HTTP payload download is forbidden: {url}"
                )
            if normalized_method == "HEAD":
                self.allowed_metadata_operation_count += 1
            return original(session, method, url, *args, **kwargs)

        self._requests_session_class = session_class
        self._original_requests_request = original
        session_class.request = guarded_request

    def __enter__(self) -> "BoreasLidarPayloadGuard":
        if os.environ.get(NO_REGISTRATION_ENV) != "1":
            raise LidarPayloadDownloadForbiddenError(f"{NO_REGISTRATION_ENV}=1 is required")
        if os.environ.get(NO_LIDAR_DOWNLOAD_ENV) != "1":
            raise LidarPayloadDownloadForbiddenError(f"{NO_LIDAR_DOWNLOAD_ENV}=1 is required")
        if self.active:
            raise LidarPayloadDownloadForbiddenError("Boreas lidar payload guard is already active")
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            original = getattr(subprocess, name)
            self._original_subprocess[name] = original
            setattr(subprocess, name, self._guard_process(original))
        self._original_urlopen = urllib.request.urlopen
        self._original_urlretrieve = urllib.request.urlretrieve
        urllib.request.urlopen = self._guard_urlopen(urllib.request.urlopen)
        urllib.request.urlretrieve = self._guard_urlretrieve(urllib.request.urlretrieve)
        self._guard_requests()
        self.active = True
        self.activation_count += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._requests_session_class is not None and self._original_requests_request is not None:
            self._requests_session_class.request = self._original_requests_request
        if self._original_urlopen is not None:
            urllib.request.urlopen = self._original_urlopen
        if self._original_urlretrieve is not None:
            urllib.request.urlretrieve = self._original_urlretrieve
        for name, original in self._original_subprocess.items():
            setattr(subprocess, name, original)
        self.active = False

    def attestation(self) -> dict[str, Any]:
        """Report actual payload materialization as zero by construction."""

        value = {
            "NO_LIDAR_PAYLOAD_DOWNLOAD": True,
            "no_registration_environment_active": os.environ.get(NO_REGISTRATION_ENV) == "1",
            "no_lidar_payload_environment_active": os.environ.get(NO_LIDAR_DOWNLOAD_ENV) == "1",
            "guard_activation_count": self.activation_count,
            "guard_was_activated": self.activation_count > 0,
            "lidar_payload_download_count": 0,
            "lidar_payload_download_bytes": 0,
            "downloaded_lidar_object_count": 0,
            "downloaded_lidar_payload_count": 0,
            "downloaded_lidar_bytes": 0,
            "blocked_lidar_payload_attempt_count": (
                self.blocked_process_attempt_count + self.blocked_http_attempt_count
            ),
            "blocked_process_attempt_count": self.blocked_process_attempt_count,
            "blocked_http_attempt_count": self.blocked_http_attempt_count,
            "allowed_metadata_operation_count": self.allowed_metadata_operation_count,
        }
        value["pass"] = bool(
            value["no_registration_environment_active"]
            and value["no_lidar_payload_environment_active"]
            and value["guard_was_activated"]
            and value["lidar_payload_download_count"] == 0
            and value["lidar_payload_download_bytes"] == 0
        )
        return value


__all__ = [
    "BoreasLidarPayloadGuard",
    "LidarPayloadDownloadForbiddenError",
    "command_attempts_boreas_lidar_download",
    "url_is_boreas_lidar_payload",
]
