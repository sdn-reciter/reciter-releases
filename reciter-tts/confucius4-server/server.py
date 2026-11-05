import hmac
import io
import os
import re
import subprocess
import tempfile
import threading
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

# ─────────────────────────────────────────────────────────────────────────────
#  Confucius4-TTS сервер для Reciter — тот же API, что у остальных серверов
#  (/tts, /voices, /health): приложение переключается сменой адреса (порт
#  8008). Голос — клонированием по wav-образцу из ./voices; расшифровка
#  образца модели не нужна.
#
#  Окружение (docker-compose.yml):
#  API_KEY          — если задан, нужен заголовок Authorization: Bearer <key>.
#  DEFAULT_SPEAKER  — голос по умолчанию (имя файла в ./voices без расширения).
#  REF_MAX_SEC      — эталон длиннее режется: модель берёт 3–15 с.
# ─────────────────────────────────────────────────────────────────────────────

SERVER_VERSION = "cf-1 (2026-10-31)"
API_KEY = os.environ.get("API_KEY", "").strip()
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices")
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "").strip()
REF_MAX_SEC = float(os.environ.get("REF_MAX_SEC", "12"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))
# Модель принимает до 1024 знаков; длинный абзац делится по предложениям.
CHUNK_CHARS = 400
REPO = "/opt/confucius"
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

os.makedirs(VOICES_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 60)
print(f"Loading Confucius4-TTS on {DEVICE}...")
# Пути в inference_config.yaml относительные («./checkpoints») — от корня репозитория.
_cwd = os.getcwd()
os.chdir(REPO)
from confuciustts.cli.inference import ConfuciusTTS  # noqa: E402

model = ConfuciusTTS(config_path=f"{REPO}/config/inference_config.yaml", device=DEVICE)
os.chdir(_cwd)
SAMPLE_RATE = int(model.sample_rate)
print(f"Sample rate: {SAMPLE_RATE}")
print(f"  Reciter Confucius4 server — SERVER_VERSION {SERVER_VERSION}")
print(f"API auth: {'ON' if API_KEY else 'OFF'}")
print("=" * 60)

app = FastAPI(title="Confucius4-TTS Server")
_gen_lock = threading.Lock()


def require_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    if not authorization or not hmac.compare_digest(authorization.strip(), f"Bearer {API_KEY}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


def safe_voice_name(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name).strip("._") or "voice"


def list_voice_names():
    return sorted({os.path.splitext(f)[0] for f in os.listdir(VOICES_DIR) if f.lower().endswith(AUDIO_EXT)})


def resolve_speaker(voice: Optional[str]) -> str:
    """Путь к эталону: запрошенный → DEFAULT_SPEAKER → первый файл. Без
    эталона модель не говорит — тогда ошибка с подсказкой."""
    for name in (voice, DEFAULT_SPEAKER):
        if not name:
            continue
        for ext in AUDIO_EXT:
            p = os.path.join(VOICES_DIR, f"{safe_voice_name(name)}{ext}")
            if os.path.isfile(p):
                return p
    names = list_voice_names()
    if not names:
        raise ValueError("Нет ни одного голоса: загрузите wav через /voices/upload (5–12 с чистой речи).")
    for ext in AUDIO_EXT:
        p = os.path.join(VOICES_DIR, f"{names[0]}{ext}")
        if os.path.isfile(p):
            return p
    raise ValueError("Голос не найден")


def trimmed_ref(path: str) -> str:
    """Эталон в WAV 16 кГц и не длиннее REF_MAX_SEC — модель берёт 3–15 с."""
    out = os.path.join(tempfile.gettempdir(), f"ref_{safe_voice_name(os.path.basename(path))}_{int(os.path.getmtime(path))}.wav")
    if not os.path.isfile(out):
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", path, "-t", str(REF_MAX_SEC),
                        "-ac", "1", "-ar", "16000", out], check=True)
    return out


# Метки ударений модель не знает: акут и «+» перед гласной убираются.
def strip_stress(text: str) -> str:
    return re.sub(r"\+(?=[аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "", text.replace("́", ""))


_SENT = re.compile(r"(?<=[.!?…])\s+")
_SPEECH = re.compile(r"[^\W_]")


def chunks(text: str):
    out, cur = [], ""
    for s in (s for s in _SENT.split(text) if _SPEECH.search(s)):
        if cur and len(cur) + len(s) + 1 > CHUNK_CHARS:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def synthesize(text: str, voice: Optional[str], lang: str) -> io.BytesIO:
    text = strip_stress((text or "").strip())[:MAX_TEXT_CHARS]
    ref = trimmed_ref(resolve_speaker(voice))
    parts = chunks(text)
    pause = np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
    audio = []
    with _gen_lock:
        for i, c in enumerate(parts):
            wav = model.generate(c, lang, ref, verbose=False)
            audio += ([pause] if i else []) + [wav.detach().float().cpu().numpy().reshape(-1)]
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(audio) if audio else np.zeros(int(SAMPLE_RATE * 0.12), np.float32),
             SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf


@app.get("/health")
async def health():
    return {"status": "ok", "server_version": SERVER_VERSION, "engine": "confucius4", "device": DEVICE,
            "sample_rate": SAMPLE_RATE, "auth": bool(API_KEY), "voices": list_voice_names(),
            "default_voice": DEFAULT_SPEAKER, "reference_wav": True}


@app.post("/tts")
async def tts(text: str = Form(...), language: str = Form("ru"), voice: Optional[str] = Form(None),
              authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    lang = (language or "ru").split("-")[0].lower()
    try:
        return StreamingResponse(synthesize(text, voice, lang), media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/voices")
async def voices(authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    return {"voices": list_voice_names(), "default": DEFAULT_SPEAKER}


@app.post("/voices/upload")
async def upload_voice(name: str = Form(...), file: UploadFile = File(...),
                       authorization: Optional[str] = Header(None)):
    """Эталон голоса: 5–12 с чистой речи; перекодируется в WAV."""
    require_auth(authorization)
    dst = os.path.join(VOICES_DIR, f"{safe_voice_name(name)}.wav")
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or "")[1] or ".bin", delete=False) as tmp:
        tmp.write(await file.read())
    try:
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", tmp.name, "-ac", "1", dst], check=True)
    except subprocess.CalledProcessError:
        return JSONResponse(status_code=400, content={"error": "не удалось прочитать аудио"})
    finally:
        os.unlink(tmp.name)
    return {"status": "ok", "voice": safe_voice_name(name)}


@app.delete("/voices/{name}")
async def delete_voice(name: str, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    gone = 0
    for ext in AUDIO_EXT:
        p = os.path.join(VOICES_DIR, f"{safe_voice_name(name)}{ext}")
        if os.path.isfile(p):
            os.remove(p)
            gone += 1
    if not gone:
        raise HTTPException(status_code=404, detail="нет такого голоса")
    return {"status": "ok"}
