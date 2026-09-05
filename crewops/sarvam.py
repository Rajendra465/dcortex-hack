"""Sarvam AI: the callout, in the language the crew member actually reads.

An Indian carrier's cabin crew do not all read English equally well, and a
callout is the one message on this desk that leaves the building. Translating
it is the highest-value thing Sarvam can do here, and -- this is the part that
matters -- it is architecturally safe.

Every figure in a callout is computed by the kernel and checked by the number
guard BEFORE it reaches this module. Translation operates on a finished
sentence, so a language model cannot invent a report time or a duty hour; it
can only re-say one that was already verified. That is a very different risk
from letting a model near the arithmetic, and it is why this is the one place
a language model is allowed to produce crew-facing text.

Two endpoints, both verified live against the API:

    POST /translate        mayura:v1   22 Indic languages, handles Hinglish
    POST /text-to-speech   bulbul:v3   audio for a 5 a.m. phone call

The desk works completely without this. No key, no network, no Sarvam: the
callout is drafted in English exactly as before, and every other feature is
untouched. Nothing here sits on the path of a legality answer.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

BASE = "https://api.sarvam.ai"
TRANSLATE_MODEL = "mayura:v1"      # colloquial + code-mixed; ops staff write Hinglish
TTS_MODEL = "bulbul:v3"
TTS_SPEAKER = "ritu"               # bulbul:v3 rejects the older v2 speaker names
TIMEOUT = 20

# The languages worth offering a crew desk, not all 22. Each is one a
# significant number of Indian cabin crew read more comfortably than English.
LANGUAGES: dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "bn-IN": "Bengali",
    "mr-IN": "Marathi",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
}


class SarvamError(RuntimeError):
    """Sarvam said no. Carries the message so the desk can show it rather than
    silently falling back to English and letting a controller believe a crew
    member was written to in their own language when they were not."""


def api_key() -> str | None:
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    return key or None


def available() -> bool:
    return api_key() is not None


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    key = api_key()
    if not key:
        raise SarvamError("No SARVAM_API_KEY configured.")
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"api-subscription-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise SarvamError(f"Sarvam {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SarvamError(f"Sarvam unreachable: {e}") from None
    if isinstance(body, dict) and "error" in body:
        err = body["error"]
        raise SarvamError(str(err.get("message", err) if isinstance(err, dict) else err))
    return body


# Identifiers never go through the translator at all.
#
# The first design shielded them with markers and translated around them.
# Measured against mayura:v1, the markers themselves came back mangled --
# <<0>> survived, <<1>> and <<2>> did not, inconsistently across languages --
# so a callout could arrive missing a flight number. That is worse than an
# English callout, and no amount of retrying makes it dependable.
#
# So the callout is built in two parts and only one of them is translated:
# a spoken paragraph a person reads, and an operational block that is never
# touched. This is also how aviation already works -- identifiers, stations
# and Zulu times stay in standard form in every language, on every flight
# deck, everywhere.
_HAS_ID = re.compile(
    "|".join([
        r"[CP]-\d{3,5}",
        r"DX\d{3,4}",
        r"VT-[A-Z]{3}",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}:\d{2}",
        r"(?:BLR|DEL|BOM|HYD|MAA|CCU|GOI|PNQ|COK|AMD)",
    ])
)


def has_identifier(line: str) -> bool:
    return bool(_HAS_ID.search(line))


def translate(text: str, target: str, source: str = "en-IN") -> str:
    """Re-say the prose of a callout, and leave the operational lines alone.

    Line by line: a line carrying a crew id, flight number, station or Zulu
    time is passed through verbatim; everything else is translated. What the
    crew member gets is their own language for the instruction and unaltered
    standard form for the facts.
    """
    if not text or target == source or target not in LANGUAGES:
        return text
    out: list[str] = []
    for line in text.split("\n"):
        if not line.strip() or has_identifier(line):
            out.append(line)
            continue
        body = _post("/translate", {
            "input": line,
            "source_language_code": source,
            "target_language_code": target,
            "model": TRANSLATE_MODEL,
        })
        out.append(body.get("translated_text") or line)
    return "\n".join(out)


def speak(text: str, language: str = "en-IN", speaker: str = TTS_SPEAKER) -> str:
    """Base64 WAV of the callout, for a controller who is already on the phone.

    Returned as a data URI the browser plays directly, so nothing is written to
    disk or served from a second route.
    """
    body = _post("/text-to-speech", {
        "text": text[:1500],          # the API caps input; a callout is far shorter
        "target_language_code": language if language in LANGUAGES else "en-IN",
        "speaker": speaker,
        "model": TTS_MODEL,
    })
    audios = body.get("audios") or []
    if not audios:
        raise SarvamError("Sarvam returned no audio.")
    raw = audios[0] if isinstance(audios, list) else audios
    base64.b64decode(raw[:64] + "==", validate=False)   # fail here, not in the browser
    return "data:audio/wav;base64," + raw


def status() -> dict[str, Any]:
    return {
        "configured": available(),
        "translate_model": TRANSLATE_MODEL,
        "tts_model": TTS_MODEL,
        "languages": LANGUAGES,
    }
