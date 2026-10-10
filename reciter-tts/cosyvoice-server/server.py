import io
import os
import re
import sys
import threading

import numpy as np
import soundfile as sf

# Метка версии — печатается при старте и отдаётся в /health.
SERVER_VERSION = "cv3-2 (2026-07-17: CV3 требует префикс <|endofprompt|>)"

from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import torch

# ─────────────────────────────────────────────────────────────────────────────
#  Fun-CosyVoice3-0.5B (Alibaba FunAudioLLM) сервер для Reciter — тот же API, что
#  у XTTS/Supertonic/VoxCPM/ESpeech (/tts, /accent, /voices, /health): порт 8007.
#
#  Почему v3, а не v2: CosyVoice2-0.5B РУССКИЙ официально НЕ поддерживал (озвучивал
#  кириллицу «по-китайски»). Fun-CosyVoice3-0.5B — 9 языков, русский среди них,
#  Apache-2.0, качество/похожесть выше v2. Класс CosyVoice3 в том же репозитории.
#
#  Особенность (как F5): нужен РЕФЕРЕНС — модель клонирует по образцу, своего
#  «встроенного» голоса у 0.5B нет. Используем cross-lingual инференс: он берёт
#  только аудио-референс (транскрипт не нужен) и синтезирует текст на его тембре.
#
#  Ударения: CosyVoice3 — нейромодель без управления ударением (метки не
#  читает). SYNTH_STRESS=strip. /accent остаётся для приложения. Управляемое
#  ударение — ESpeech (F5).
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_KEY", "").strip()
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models/Fun-CosyVoice3-0.5B").strip()
COSY_REPO = os.environ.get("COSY_REPO", "/app/CosyVoice").strip()
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices").strip()
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "").strip()
USE_RUACCENT = os.environ.get("USE_RUACCENT", "1").strip().lower() in ("1", "true", "yes")
STRESS_MODE = os.environ.get("COSY_STRESS_MODE", "mark").strip().lower()
SYNTH_STRESS = os.environ.get("COSY_SYNTH_STRESS", "strip").strip().lower()
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))
SENT_PAUSE_MS = float(os.environ.get("COSY_SENT_PAUSE_MS", "180"))
CHUNK_CHARS = int(os.environ.get("COSY_CHUNK_CHARS", "220"))
# CosyVoice извлекает speech-token только из эталона ≤30 с (жёсткий assert во
# frontend). Длинные образцы обрезаем; ~18 с — оптимум качества клона.
PROMPT_MAX_SEC = float(os.environ.get("COSY_PROMPT_MAX_SEC", "18"))
PROMPT_CACHE_DIR = os.environ.get("COSY_PROMPT_CACHE_DIR", "/tmp/cosy-prompt-cache").strip()
# CosyVoice3 требует служебный токен <|endofprompt|> во входе LLM (иначе assert
# «<|endofprompt|> not detected»). Всё ДО токена — системная инструкция (не
# озвучивается), после — синтезируемый текст. Наличие <|...|> к тому же
# отключает китайский нормализатор и разбиение (text_normalize отдаёт строку
# целиком) — для русского это и нужно. Системный промпт можно переопределить.
CV3_SYS_PROMPT = os.environ.get("COSY_SYS_PROMPT", "You are a helpful assistant.")
_END_OF_PROMPT = "<|endofprompt|>"

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)

app = FastAPI(title="CosyVoice3 Server")

_dev_env = os.environ.get("DEVICE", "auto").strip().lower()
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _dev_env == "auto" else _dev_env
print("=" * 60)
print(f"Loading Fun-CosyVoice3 ({MODEL_DIR}) on {DEVICE}...")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Matcha-TTS (third_party подмодуль CosyVoice) обязан быть в PYTHONPATH.
sys.path.append(os.path.join(COSY_REPO, "third_party", "Matcha-TTS"))
sys.path.append(COSY_REPO)

from cosyvoice.cli.cosyvoice import CosyVoice3

# CosyVoice3.__init__ не принимает load_jit (в отличие от CosyVoice2).
cosyvoice = CosyVoice3(MODEL_DIR, load_trt=False, fp16=(DEVICE == "cuda"))
SAMPLE_RATE = int(getattr(cosyvoice, "sample_rate", 24000))

# ── Ударения (RUAccent) — контракт /accent как у остальных ───────────────────
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


# ── Голоса (референс обязателен, транскрипт НЕ нужен — cross-lingual) ─────────
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
    for name in (voice, DEFAULT_SPEAKER):
        if not name:
            continue
        for ext in _AUDIO_EXTS:
            p = os.path.join(VOICES_DIR, f"{safe_voice_name(name)}{ext}")
            if os.path.isfile(p):
                return p
    names = list_voice_names()
    if names:
        for ext in _AUDIO_EXTS:
            p = os.path.join(VOICES_DIR, f"{names[0]}{ext}")
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


_gen_lock = threading.Lock()
_prompt_lock = threading.Lock()


def _capped_prompt_path(path: str) -> str:
    # CosyVoice извлекает speech-token только из эталона ≤30 с — длинные образцы
    # роняют синтез («audio longer than 30s»). Готовим укороченную (mono, первые
    # PROMPT_MAX_SEC) копию и передаём её ПУТЬ. Кэш по (имя+mtime+лимит).
    try:
        info = sf.info(path)
        if info.frames / float(info.samplerate) <= PROMPT_MAX_SEC:
            return path
    except Exception:
        return path

    mtime = int(os.path.getmtime(path))
    key = f"{safe_voice_name(os.path.basename(path))}_{mtime}_{int(PROMPT_MAX_SEC)}s.wav"
    dst = os.path.join(PROMPT_CACHE_DIR, key)
    with _prompt_lock:
        if os.path.isfile(dst):
            return dst
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)                      # → mono
        data = data[: int(sr * PROMPT_MAX_SEC)]       # первые N секунд
        tmp = dst + ".part"
        sf.write(tmp, data, sr, format="WAV")
        os.replace(tmp, dst)
        print(f"prompt capped: {os.path.basename(path)} -> {PROMPT_MAX_SEC}s", flush=True)
    return dst


def _generate_one(text: str, prompt_path: str) -> np.ndarray:
    # CosyVoice сам загружает эталон из ФАЙЛА (frontend вызывает load_wav) —
    # передаём ПУТЬ, а не тензор (иначе torchaudio 2.9/torchcodec падает
    # «video_tensor must be kUInt8» на попытке декодировать тензор как файл).
    prompt_path = _capped_prompt_path(prompt_path)
    # Префикс с <|endofprompt|> — обязателен для CosyVoice3 (см. CV3_SYS_PROMPT).
    text = f"{CV3_SYS_PROMPT}{_END_OF_PROMPT}{text}"
    parts = []
    with _gen_lock:
        try:
            for out in cosyvoice.inference_cross_lingual(text, prompt_path, stream=False):
                w = out["tts_speech"]
                w = w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
                parts.append(w.astype(np.float32).flatten())
        finally:
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def synthesize(text: str, voice: Optional[str]) -> io.BytesIO:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    text = _apply_stress_format(text, SYNTH_STRESS)   # метки CosyVoice не читает

    speaker = resolve_speaker(voice)
    if speaker is None:
        raise HTTPException(
            status_code=503,
            detail="Нет ни одного эталона голоса. CosyVoice3 клонирует по образцу: "
                   "загрузите wav через /voices/upload (6–15 с чистой речи).",
        )
    chunks = _chunks(text)
    if not chunks:
        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32), SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return buf

    pause = np.zeros(int(SAMPLE_RATE * SENT_PAUSE_MS / 1000.0), dtype=np.float32)
    audio = _generate_one(chunks[0], speaker)
    for c in chunks[1:]:
        audio = np.concatenate([audio, pause, _generate_one(c, speaker)])

    print(f"synth[{SYNTH_STRESS}] chunks={len(chunks)} -> {text[:120]}", flush=True)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf


# ── HTTP-контракт (идентичен остальным серверам) ─────────────────────────────

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
    return {"voices": names, "default": DEFAULT_SPEAKER or (names[0] if names else "")}


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
        "engine": "cosyvoice3",
        "model_dir": MODEL_DIR,
        "server_version": SERVER_VERSION,
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
        "synth_stress": SYNTH_STRESS,
        "ruaccent": _accentizer is not None,
        "voices": list_voice_names(),
    }


print("=" * 60)
print(f"  Reciter Fun-CosyVoice3 server — SERVER_VERSION {SERVER_VERSION}")
print("=" * 60)
print(f"Device: {DEVICE}; sr={SAMPLE_RATE}, synth_stress={SYNTH_STRESS}")
print(f"API auth: {'ON' if API_KEY else 'OFF (LAN only!)'}")
print(f"Voices dir: {VOICES_DIR} -> {list_voice_names()} (референс ОБЯЗАТЕЛЕН)")
print(f"RUAccent: {'ON' if _accentizer is not None else 'OFF'}")
