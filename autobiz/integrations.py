"""Public, keyless API integrations and JSON-to-table conversion."""

from __future__ import annotations

import json
import re
import ssl
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import certifi
import pandas as pd


MAX_RESPONSE_BYTES = 10 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 15
_RECORD_KEYS = ("data", "results", "records", "items", "holidays")
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _validate_url(url: str) -> str:
    candidate = url.strip()
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("URL API harus menggunakan HTTPS.")
    if parsed.username or parsed.password:
        raise ValueError("Jangan menaruh username atau kata sandi di URL.")
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost", "127.0.0.1", "::1"
    }:
        raise ValueError("HTTP tanpa enkripsi hanya diizinkan untuk localhost.")
    return candidate


def _request_json(url: str) -> object:
    safe_url = _validate_url(url)
    request = Request(
        safe_url,
        headers={"Accept": "application/json", "User-Agent": "AutoBizData/1.0"},
    )
    try:
        with urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
            context=_SSL_CONTEXT,
        ) as response:
            _validate_url(response.geturl())
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"API mengembalikan HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Tidak dapat menghubungi API: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Permintaan API melewati batas waktu.") from exc

    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Respons API melebihi batas 10 MB.")
    try:
        return json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Respons API bukan JSON UTF-8 yang valid.") from exc


def json_to_dataframe(payload: object) -> pd.DataFrame:
    records: object = payload
    if isinstance(payload, dict):
        for key in _RECORD_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, list) and all(
                isinstance(item, dict) for item in candidate
            ):
                records = candidate
                break

    if isinstance(records, list) and all(
        isinstance(item, dict) for item in records
    ):
        frame = pd.json_normalize(records, sep=".")
    elif isinstance(records, dict):
        values = list(records.values())
        if values and all(not isinstance(value, (dict, list)) for value in values):
            frame = pd.DataFrame([records])
        elif values and all(isinstance(value, list) for value in values):
            try:
                frame = pd.DataFrame(records)
            except ValueError as exc:
                raise ValueError(
                    "Kolom berbentuk array dari API memiliki panjang berbeda."
                ) from exc
        else:
            raise ValueError(
                "JSON API harus berupa daftar objek atau objek berisi records/data."
            )
    else:
        raise ValueError(
            "JSON API harus berupa daftar objek atau objek berisi records/data."
        )

    if frame.empty or len(frame.columns) == 0:
        raise ValueError("API tidak mengembalikan baris data untuk diimpor.")
    return frame


def fetch_json_dataframe(url: str) -> pd.DataFrame:
    return json_to_dataframe(_request_json(url))


def public_holiday_url(country_code: str, year: int) -> str:
    code = country_code.strip().upper()
    if not re.fullmatch(r"[A-Z]{2}", code):
        raise ValueError("Kode negara harus dua huruf, misalnya ID.")
    if not 2000 <= year <= 2100:
        raise ValueError("Tahun kalender harus antara 2000 dan 2100.")
    return f"https://date.nager.at/api/v4/Holidays/{code}/{year}"


def fetch_public_holidays(country_code: str, year: int) -> pd.DataFrame:
    return fetch_json_dataframe(public_holiday_url(country_code, year))
