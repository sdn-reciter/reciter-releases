import io
import os
import re
import threading

import numpy as np
import soundfile as sf

# Метка версии — печатается при старте и отдаётся в /health.
SERVER_VERSION = "cb-2 (2026-07-16: ударения U+0301 через RUAccent — как russian_text_stresser)"

from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import torch

# ─────────────────────────────────────────────────────────────────────────────
#  Chatterbox Multilingual (Resemble AI) сервер для Reciter — тот же API, что у
#  XTTS/Supertonic/VoxCPM/ESpeech (/tts, /accent, /voices, /health): приложение
#  переключается сменой адреса (порт 8006 в compose).
#
#  Почему Chatterbox: ~0.5B, лицензия MIT (в отличие от XTTS-v2/CPML), 23 языка
#  включая русский, zero-shot клонирование голоса, контроль экспрессии
#  (exaggeration/cfg_weight), активный репозиторий. Есть встроенный голос —
#  референс НЕ обязателен (в отличие от F5/CosyVoice).
#
#  Ударения: Chatterbox ОБУЧЕН на русском С ударениями — его встроенный
#  russian_text_stresser ставит их в формате U+0301 (комбинируемый акут).
#  Этот пакет конфликтует по зависимостям (spacy 3.6 vs gradio) и в образе не
#  ставится, поэтому ударения проставляем СВОИМ RUAccent в ТОМ ЖЕ формате
#  (U+0301) — модель их читает (без russian_text_stresser текст проходит в
#  модель как есть). Управляется CB_SYNTH_STRESS (mark по умолчанию).
#
#  Конфигурация через окружение (docker-compose.yml)
#  API_KEY            — если задан, нужен заголовок Authorization: Bearer <key>.
#  DEVICE             — auto (по умолчанию) | cuda | cpu.
#  VOICES_DIR         — каталог эталонов (.wav и т.п.) для клонирования.
#  DEFAULT_SPEAKER    — голос по умолчанию; пусто → встроенный голос модели.
#  LANGUAGE_ID        — язык синтеза (ru по умолчанию).
#  CB_EXAGGERATION    — экспрессия (0.25–1.0; 0.5 — нейтрально).
#  CB_CFG_WEIGHT      — сила следования тексту/темпу (0.3–0.7).
#  USE_RUACCENT       — "1": ударения в /accent (для приложения).
#  MAX_TEXT_CHARS     — предохранитель от гигантских запросов.
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_KEY", "").strip()
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices").strip()
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "").strip()
LANGUAGE_ID = os.environ.get("LANGUAGE_ID", "ru").strip().lower()
EXAGGERATION = float(os.environ.get("CB_EXAGGERATION", "0.5"))
CFG_WEIGHT = float(os.environ.get("CB_CFG_WEIGHT", "0.5"))
USE_RUACCENT = os.environ.get("USE_RUACCENT", "1").strip().lower() in ("1", "true", "yes")
STRESS_MODE = os.environ.get("CB_STRESS_MODE", "mark").strip().lower()
# Chatterbox обучен на U+0301 (его russian_text_stresser). mark по умолчанию —
# ставим ударения СВОИМ RUAccent в этом формате. strip — без ударений.
SYNTH_STRESS = os.environ.get("CB_SYNTH_STRESS", "mark").strip().lower()
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))
SENT_PAUSE_MS = float(os.environ.get("CB_SENT_PAUSE_MS", "180"))
# Chatterbox стабилен на кусках ~ до одного длинного предложения; длинный
# абзац дробим по границам предложений.
CHUNK_CHARS = int(os.environ.get("CB_CHUNK_CHARS", "300"))

os.makedirs(VOICES_DIR, exist_ok=True)

app = FastAPI(title="Chatterbox Multilingual Server")

_dev_env = os.environ.get("DEVICE", "auto").strip().lower()
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _dev_env == "auto" else _dev_env
print("=" * 60)
print(f"Loading Chatterbox Multilingual on {DEVICE}...")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
SAMPLE_RATE = int(getattr(model, "sr", 24000))

# ── Ударения (RUAccent) — тот же контракт /accent, что у остальных ───────────
_STRESS = "́"
_RU_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
_MARKED_VOWEL = re.compile(f"([{''.join(sorted(_RU_VOWELS))}])" + _STRESS)

_accentizer = None
if USE_RUACCENT:
    try:
        from ruaccent import RUAccent
        _accentizer = RUAccent()
        _accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True)
        print(f"RUAccent loaded (turbo3.1); /accent={STRESS_MODE}, synth={SYNTH_STRESS}.")
    except Exception as e:
        print(f"RUAccent NOT available ({e}); accents disabled.")
        _accentizer = None


def _plus_to_mark(text: str) -> str:
    if "+" not in text:
        return text
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "+":
            if i + 1 < n and text[i + 1] in _RU_VOWELS:
                out.append(text[i + 1]); out.append(_STRESS); i += 2; continue
            if out and out[-1] in _RU_VOWELS:
                out.append(_STRESS); i += 1; continue
            out.append(ch); i += 1
            continue
        out.append(ch); i += 1
    return "".join(out)


def _apply_stress_format(text: str, mode: str) -> str:
    t = _plus_to_mark(text)
    if mode == "strip":
        return t.replace(_STRESS, "")
    if mode == "plus":
        return _MARKED_VOWEL.sub(r"+\1", t)
    return t


def accentize(text: str) -> str:
    if _accentizer is None:
        return text
    try:
        return _apply_stress_format(_accentizer.process_all(text), STRESS_MODE)
    except Exception as e:
        print(f"RUAccent error: {e}")
        return text


def _stress_for_synth(text: str) -> str:
    """Текст для модели. Chatterbox обучен на U+0301 (его russian_text_stresser).
    В mark/plus-режиме ставим ударения тем же RUAccent: если клиент уже прислал
    метки (U+0301 из словаря приложения или «+») — только нормализуем формат,
    иначе размечаем через RUAccent. strip — снимаем всё."""
    if SYNTH_STRESS == "strip":
        return _apply_stress_format(text, "strip")
    if _STRESS in text or "+" in text:
        return _apply_stress_format(text, SYNTH_STRESS)
    if _accentizer is not None:
        try:
            return _apply_stress_format(_accentizer.process_all(text), SYNTH_STRESS)
        except Exception as e:
            print(f"RUAccent synth error: {e}")
    return _apply_stress_format(text, SYNTH_STRESS)


# ── Голоса (эталоны для клонирования; необязательны — есть встроенный) ────────
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_\-]+")
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def safe_voice_name(name: str) -> str:
    slug = _SAFE_NAME.sub("_", (name or "").strip())[:64]
    return slug or "voice"


def list_voice_names():
    try:
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(VOICES_DIR)
            if f.lower().endswith(_AUDIO_EXTS)
        )
    except FileNotFoundError:
        return []


def resolve_speaker(voice: Optional[str]) -> Optional[str]:
    """Путь к эталону или None (тогда встроенный голос Chatterbox)."""
    for name in (voice, DEFAULT_SPEAKER):
        if not name:
            continue
        for ext in _AUDIO_EXTS:
            p = os.path.join(VOICES_DIR, f"{safe_voice_name(name)}{ext}")
            if os.path.isfile(p):
                return p
    return None


# ── Синтез ────────────────────────────────────────────────────────────────────
_HAS_SPEECH = re.compile(r"[^\W_]", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _chunks(text: str):
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip() and _HAS_SPEECH.search(s)]
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > CHUNK_CHARS:
            out.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out or ([text] if _HAS_SPEECH.search(text) else [])


# Приложение предрендерит абзацы параллельно — генерация строго по одной.
_gen_lock = threading.Lock()


def _generate_one(text: str, ref_path: Optional[str]) -> np.ndarray:
    with _gen_lock:
        try:
            kwargs = dict(language_id=LANGUAGE_ID,
                          exaggeration=EXAGGERATION, cfg_weight=CFG_WEIGHT)
            if ref_path:
                kwargs["audio_prompt_path"] = ref_path
            wav = model.generate(text, **kwargs)
        finally:
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    return arr.astype(np.float32).flatten()


def synthesize(text: str, voice: Optional[str]) -> io.BytesIO:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    # Ставим ударения U+0301 (свой RUAccent) — модель на них обучена.
    text = _stress_for_synth(text)

    ref = resolve_speaker(voice)
    chunks = _chunks(text)
    if not chunks:
        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32), SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return buf

    pause = np.zeros(int(SAMPLE_RATE * SENT_PAUSE_MS / 1000.0), dtype=np.float32)
    audio = _generate_one(chunks[0], ref)
    for c in chunks[1:]:
        audio = np.concatenate([audio, pause, _generate_one(c, ref)])

    print(f"synth[{SYNTH_STRESS}] lang={LANGUAGE_ID} ref={bool(ref)} chunks={len(chunks)} -> {text[:120]}",
          flush=True)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf


# ── HTTP-контракт (идентичен XTTS/Supertonic/VoxCPM/ESpeech) ─────────────────

def require_auth(authorization: Optional[str]):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


class SpeechRequest(BaseModel):
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    language: Optional[str] = "ru"


class AccentRequest(BaseModel):
    text: str


@app.post("/accent")
async def accent_endpoint(request: AccentRequest, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    return {"text": accentize(request.text or "")}


@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    language: str = Form("ru"),
    voice: str = Form(""),
    authorization: Optional[str] = Header(None),
):
    require_auth(authorization)
    try:
        return StreamingResponse(synthesize(text, voice or None), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)[:300]})


@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    try:
        return StreamingResponse(synthesize(request.input, request.voice), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:300]})


@app.get("/voices")
async def voices_endpoint(authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    names = list_voice_names()
    # Встроенный голос доступен всегда (референс не обязателен).
    return {"voices": names, "default": DEFAULT_SPEAKER or ""}


@app.post("/voices/upload")
async def upload_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    require_auth(authorization)
    slug = safe_voice_name(name)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _AUDIO_EXTS:
        ext = ".wav"
    dest = os.path.join(VOICES_DIR, slug + ext)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "voice": slug}


@app.delete("/voices/{name}")
async def delete_voice(name: str, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    slug = safe_voice_name(name)
    deleted = False
    for ext in _AUDIO_EXTS:
        p = os.path.join(VOICES_DIR, slug + ext)
        if os.path.isfile(p):
            os.remove(p)
            deleted = True
    if not deleted:
        raise HTTPException(status_code=404, detail="voice not found")
    return {"ok": True}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "chatterbox-multilingual",
        "server_version": SERVER_VERSION,
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
        "language_id": LANGUAGE_ID,
        "synth_stress": SYNTH_STRESS,
        "ruaccent": _accentizer is not None,
        "voices": list_voice_names(),
    }


print("=" * 60)
print(f"  Reciter Chatterbox server — SERVER_VERSION {SERVER_VERSION}")
print("=" * 60)
print(f"Device: {DEVICE}; sr={SAMPLE_RATE}, lang={LANGUAGE_ID}, synth_stress={SYNTH_STRESS}")
print("Ударения U+0301 ставит свой RUAccent (формат russian_text_stresser, "
      "который в образ не ставится из-за конфликта зависимостей).")
print(f"API auth: {'ON' if API_KEY else 'OFF (LAN only!)'}")
print(f"Voices dir: {VOICES_DIR} -> {list_voice_names()} (референс необязателен)")
