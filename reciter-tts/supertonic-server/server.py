import io
import os
import re
import tempfile

import numpy as np
import soundfile as sf

# Метка версии server.py — печатается при старте и отдаётся в /health,
# чтобы за секунду проверить, что в контейнере крутится свежий файл.
SERVER_VERSION = "st-4 (2026-07-16: лог текста синтеза; Supertonic G2P-free — метки ударений не действуют)"

from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from supertonic import TTS

# ─────────────────────────────────────────────────────────────────────────────
#  Supertonic-3 сервер для Reciter — тот же API, что у XTTS-сервера
#  (/tts, /accent, /voices, /health), поэтому приложение переключается на него
#  простой сменой адреса сервера (порт 8003 в compose).
#
#  Отличия от XTTS:
#   • Голоса — пресеты Supertonic (F1–F5, M1–M5) + JSON-стили из Voice Builder
#    (кладутся в STYLES_DIR). Клонирования по wav-образцу НЕТ — /voices/upload
#    принимает только .json и объясняет это в ошибке.
#   • Модель лёгкая (~99M, ONNX) — синтез быстрее реального времени даже на CPU.
# ─────────────────────────────────────────────────────────────────────────────
#  Конфигурация через окружение (docker-compose.yml)
#  API_KEY            — если задан, нужен заголовок Authorization: Bearer <key>.
#  STYLES_DIR         — каталог с JSON-стилями голосов из Voice Builder.
#  DEFAULT_VOICE      — голос по умолчанию (пресет или имя JSON без расширения).
#  USE_RUACCENT       — "1": проставлять ударения в /accent (для приложения).
#  SUPERTONIC_STEPS   — качество синтеза (5–12, больше = лучше и медленнее).
#  SUPERTONIC_SPEED   — базовая скорость речи (скорость приложения умножается
#                       на неё на стороне клиента при воспроизведении).
#  SUPERTONIC_SYNTH_STRESS — что скармливать модели: strip (по умолчанию) |
#                       mark | plus. ВАЖНО: Supertonic работает по сырым
#                       символам без G2P и без фонетической разметки — метки
#                       ударений он НЕ читает. strip тут единственно верный:
#                       mark/plus только подсовывают модели неизвестный символ.
#                       Внешнее управление ударением (словарь/`/accent`) на
#                       Supertonic не действует — для этого нужен ESpeech (F5).
#  MAX_TEXT_CHARS     — предохранитель от гигантских запросов.
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("API_KEY", "").strip()
STYLES_DIR = os.environ.get("STYLES_DIR", "/app/styles").strip()
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "F1").strip()
USE_RUACCENT = os.environ.get("USE_RUACCENT", "1").strip().lower() in ("1", "true", "yes")
STEPS = max(1, min(16, int(os.environ.get("SUPERTONIC_STEPS", "8"))))
BASE_SPEED = float(os.environ.get("SUPERTONIC_SPEED", "1.0"))
STRESS_MODE = os.environ.get("SUPERTONIC_STRESS_MODE", "mark").strip().lower()
SYNTH_STRESS = os.environ.get("SUPERTONIC_SYNTH_STRESS", "strip").strip().lower()
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))
# auto|gpu|cpu. auto = GPU, если CUDAExecutionProvider доступен, иначе CPU.
SUPERTONIC_DEVICE = os.environ.get("SUPERTONIC_DEVICE", "auto").strip().lower()

os.makedirs(STYLES_DIR, exist_ok=True)

app = FastAPI(title="Supertonic-3 Server")

print("Loading Supertonic-3 model...")
try:
    import onnxruntime
    _AVAILABLE_PROVIDERS = onnxruntime.get_available_providers()
    print(f"ONNX Runtime providers: {_AVAILABLE_PROVIDERS}")
except Exception as e:
    _AVAILABLE_PROVIDERS = []
    print(f"onnxruntime info unavailable: {e}")

# Supertonic жёстко задаёт DEFAULT_ONNX_PROVIDERS=["CPUExecutionProvider"]
# в supertonic.config и импортирует это имя в supertonic.loader, где оно и
# читается при создании InferenceSession. Поэтому GPU не включается сам —
# переопределяем список провайдеров ДО конструирования TTS. Патчим оба
# модуля: loader уже связал имя своим `from .config import ...`.
_want_gpu = SUPERTONIC_DEVICE in ("auto", "gpu") and "CUDAExecutionProvider" in _AVAILABLE_PROVIDERS
if SUPERTONIC_DEVICE == "gpu" and "CUDAExecutionProvider" not in _AVAILABLE_PROVIDERS:
    print("SUPERTONIC_DEVICE=gpu, но CUDAExecutionProvider недоступен — падаем на CPU. "
          "Проверь, что установлен onnxruntime-gpu и контейнер видит GPU.")
if _want_gpu:
    _providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        import supertonic.config as _sc
        import supertonic.loader as _sl
        _sc.DEFAULT_ONNX_PROVIDERS = _providers
        _sl.DEFAULT_ONNX_PROVIDERS = _providers
        print(f"Supertonic providers -> {_providers} (GPU)")
    except Exception as e:
        print(f"Не удалось переопределить провайдеры Supertonic ({e}); остаёмся на CPU.")

# auto_download=True: при сборке образа ассеты уже прогреты (см. Dockerfile),
# поэтому на старте контейнера скачивания нет.
tts = TTS(auto_download=True)

BUILTIN_VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]

# ── Ударения (RUAccent) — тот же контракт /accent, что у XTTS-сервера ────────
_STRESS = "́"                       # комбинируемый акут (ударение)
_RU_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
_MARKED_VOWEL = re.compile(f"([{''.join(sorted(_RU_VOWELS))}])" + _STRESS)

_accentizer = None
if USE_RUACCENT:
    try:
        from ruaccent import RUAccent
        _accentizer = RUAccent()
        _accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True)
        print(f"RUAccent loaded (turbo3.1); /accent={STRESS_MODE}, synth={SYNTH_STRESS}.")
    except Exception as e:
        print(f"RUAccent NOT available ({e}); accents disabled.")
        _accentizer = None


def _plus_to_mark(text: str) -> str:
    """«+»-метки (перед гласной) → U+0301 после гласной; «+» вне гласных не трогается."""
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
    """Любая разметка ударений («+»/U+0301) → формат [mode]: mark|plus|strip."""
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


# ── Голоса ────────────────────────────────────────────────────────────────────
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_\-]+")


def safe_voice_name(name: str) -> str:
    slug = _SAFE_NAME.sub("_", (name or "").strip())[:64]
    return slug or "voice"


def list_style_names():
    try:
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(STYLES_DIR)
            if f.lower().endswith(".json")
        )
    except FileNotFoundError:
        return []


def list_voice_names():
    return BUILTIN_VOICES + list_style_names()


def resolve_style(voice: Optional[str]):
    """Стиль голоса: JSON из STYLES_DIR → пресет → DEFAULT_VOICE → F1."""
    for name in (voice, DEFAULT_VOICE, "F1"):
        if not name:
            continue
        p = os.path.join(STYLES_DIR, f"{safe_voice_name(name)}.json")
        if os.path.isfile(p):
            return tts.get_voice_style_from_path(p)
        if name in BUILTIN_VOICES:
            return tts.get_voice_style(voice_name=name)
    return tts.get_voice_style(voice_name="F1")


print("=" * 60)
print(f"  Reciter Supertonic server — SERVER_VERSION {SERVER_VERSION}")
print("=" * 60)
print(f"API auth: {'ON' if API_KEY else 'OFF (LAN only!)'}")
print(f"Voices: builtin {BUILTIN_VOICES} + styles {list_style_names() or 'EMPTY'}")
print(f"RUAccent: {'ON' if _accentizer else 'OFF'}; steps={STEPS}, base speed={BASE_SPEED}")
print(f"synth_stress={SYNTH_STRESS} (Supertonic G2P-free: метки ударений в синтезе не действуют; "
      f"для управления ударением используйте ESpeech)")


def require_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    import hmac
    expected = f"Bearer {API_KEY}"
    if not authorization or not hmac.compare_digest(authorization.strip(), expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


# «Речевой» символ — буква/цифра. Куски без них (одни знаки препинания) в
# модель не шлём: мультиязычные TTS на голой пунктуации галлюцинируют.
_HAS_SPEECH = re.compile(r"[^\W_]", re.UNICODE)


def _silence_wav(seconds: float = 0.12) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(24000 * seconds), dtype=np.float32), 24000, format="WAV")
    buf.seek(0)
    return buf


def synthesize(text: str, voice: Optional[str], language: str = "ru") -> io.BytesIO:
    raw = (text or "").strip()
    text = _apply_stress_format(raw, SYNTH_STRESS)
    # Лог того, что реально уходит в модель (аналог gen_text у ESpeech). Если во
    # входе были метки (U+0301/«+»), а synth_stress=strip — видно, что они
    # срезаны. Внимание: сам Supertonic работает по СЫРЫМ символам без G2P и
    # без фонетической разметки — метки ударений он не читает в принципе (ни
    # strip, ни mark/plus не меняют ударение; mark/plus лишь подсовывают модели
    # неизвестный символ). Управлять ударением через словарь тут нельзя — для
    # этого есть ESpeech (F5), он stress-aware.
    _had_marks = ("+" in raw) or (_STRESS in raw)
    print(f"synth[{SYNTH_STRESS}] marks_in={_had_marks} -> {text[:200]}", flush=True)
    if not text:
        raise ValueError("Empty text")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    if not _HAS_SPEECH.search(text):
        return _silence_wav()

    style = resolve_style(voice)
    lang = (language or "ru").split("-")[0].lower()
    wav, _duration = tts.synthesize(
        text=text,
        voice_style=style,
        lang=lang,
        total_steps=STEPS,
        speed=BASE_SPEED,
    )
    # Частоту дискретизации знает сама библиотека — сохраняем её save_audio
    # во временный WAV и стримим байты как есть.
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        tts.save_audio(wav, path)
        with open(path, "rb") as f:
            data = f.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return io.BytesIO(data)


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


@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    try:
        audio_buf = synthesize(request.input, request.voice, request.language or "ru")
        return StreamingResponse(audio_buf, media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    language: str = Form("ru"),
    voice: Optional[str] = Form(None),
    accent: Optional[bool] = Form(False),
    speaker_wav: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(None),
):
    require_auth(authorization)
    if accent:
        text = accentize(text)
    # speaker_wav принимается для совместимости сигнатуры, но клонирование по
    # аудио у Supertonic нет — параметр игнорируется.
    try:
        audio_buf = synthesize(text, voice, language)
        return StreamingResponse(audio_buf, media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/voices")
async def list_voices(authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    return {"voices": list_voice_names(), "default": DEFAULT_VOICE}


@app.post("/voices/upload")
async def upload_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """Принимает ТОЛЬКО .json-стиль из Voice Builder (https://supertone.ai).
    Клонирования по wav-образцу у Supertonic нет — на аудиофайл отвечаем
    понятной ошибкой, чтобы менеджер голосов в приложении её показал."""
    require_auth(authorization)
    fname = (file.filename or "").lower()
    if not fname.endswith(".json"):
        return JSONResponse(status_code=400, content={
            "error": "Supertonic не клонирует голос по аудио. Создайте стиль в "
                     "Voice Builder (supertone.ai) и загрузите полученный .json."
        })
    data = await file.read()
    try:
        import json
        json.loads(data.decode("utf-8"))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Некорректный JSON-стиль."})
    dest = os.path.join(STYLES_DIR, f"{safe_voice_name(name)}.json")
    with open(dest, "wb") as f:
        f.write(data)
    return {"ok": True, "name": safe_voice_name(name), "voices": list_voice_names()}


@app.delete("/voices/{name}")
async def delete_voice(name: str, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    # Удаляются только пользовательские JSON-стили; пресеты неудаляемы.
    p = os.path.join(STYLES_DIR, f"{safe_voice_name(name)}.json")
    removed = False
    if os.path.isfile(p):
        os.unlink(p)
        removed = True
    return {"ok": removed, "voices": list_voice_names()}


@app.get("/health")
async def health():
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
    except Exception:
        providers = []
    return {
        "status": "ok",
        "server_version": SERVER_VERSION,
        "engine": "supertonic-3",
        "onnx_providers": providers,
        "auth": bool(API_KEY),
        "ruaccent": bool(_accentizer),
        "accent_response_mode": STRESS_MODE,
        "synth_stress": SYNTH_STRESS,
        "steps": STEPS,
        "voices": list_voice_names(),
        "default_voice": DEFAULT_VOICE,
    }
