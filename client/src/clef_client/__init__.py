"""Python client for clef-server (Jev/SystemOne ``POST /v1/systemone``)."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

__all__ = ["ClefClient", "ClefError", "choice", "encode_image", "noul", "score"]

DEFAULT_URL = "http://localhost:8000"


class ClefError(RuntimeError):
    """The server rejected the request or could not be reached."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def noul(instructions: str | None = None, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    """A true/false question. The answer is the probability of true."""
    question: dict[str, Any] = {"type": "noul"}
    if instructions:
        question["instructions"] = instructions
    if true or false:
        question["criteria"] = {key: value for key, value in (("true", true), ("false", false)) if value}
    return question


def choice(criteria: Mapping[str, str], instructions: str | None = None) -> dict[str, Any]:
    """Pick one of named options. ``criteria`` maps option ID to its description."""
    question: dict[str, Any] = {"type": "choice", "criteria": dict(criteria)}
    if instructions:
        question["instructions"] = instructions
    return question


def score(criteria: Sequence[str], instructions: str | None = None) -> dict[str, Any]:
    """Ordered options, indexed from 0. The answer includes the expected score."""
    question: dict[str, Any] = {"type": "score", "criteria": list(criteria)}
    if instructions:
        question["instructions"] = instructions
    return question


def encode_image(image: str | Path | bytes) -> str:
    """Turn a local path or raw bytes into base64. http(s) URLs and data URIs pass through unchanged."""
    if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
        return image
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    return base64.b64encode(data).decode("ascii")


class ClefClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 120.0, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def systemone(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        images: Sequence[str | Path | bytes] | None = None,
        videos: Sequence[Sequence[str | Path | bytes]] | None = None,
        model: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Send a SystemOne request and return the full response (``model``, ``answers``, ``usage``)."""
        body: dict[str, Any] = {"state": state, "questions": dict(questions), **extra}
        if model:
            body["model"] = model
        if images:
            body["images"] = [encode_image(image) for image in images]
        if videos:
            body["videos"] = [[encode_image(frame) for frame in video] for video in videos]
        return self._request("POST", "/v1/systemone", json=body)

    def answers(self, state: Any, questions: Mapping[str, Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Like :meth:`systemone`, but return only ``answers``."""
        return self.systemone(state, questions, **kwargs)["answers"]

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.request(method, self.base_url + path, timeout=self.timeout, **kwargs)
        except requests.RequestException as error:
            raise ClefError(f"cannot reach {self.base_url}: {error}") from error
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ClefError(f"{response.status_code}: {detail}", response.status_code)
        return response.json()
