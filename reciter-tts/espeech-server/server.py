import io
import os
import re
import glob
import tempfile
import threading

import numpy as np
import soundfile as sf

# Метка версии server.py — печатается при старте и отдаётся в /health.
SERVER_VERSION = "es-4 (2026-07-15: чанки ≤160 симв. — F5 глотала слова на длинных кусках)"

from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import torch

# ─────────────────────────────────────────────────────────────────────────────
#  ESpeech (русский файнтюн F5-TTS) сервер для Reciter — тот же API, что у
#  XTTS/Supertonic/VoxCPM (/tts, /accent, /voices, /health): приложение
#  переключается сменой адреса (порт 8005 в compose).
#
#  Почему ESpeech: ~0.3B параметров (в ~6 раз легче VoxCPM2), Apache 2.0,
#  обучен на 4000+ ч русской речи И на ударениях в формате «+ перед гласной»
#  — том самом, который ставит RUAccent. Поэтому по умолчанию сервер сам
#  проставляет ударения перед синтезом (SYNTH_STRESS=plus).
#
#  Особенность F5: референс обязателен ВСЕГДА (без него модель не работает)
#  и ему нужен ТРАНСКРИПТ. Транскрипт берётся из <voice>.txt рядом с wav;
#  если файла нет — распознаётся faster-whisper (CPU) один раз и кешируется
#  в тот же .txt.
#
#  Конфигурация через окружение (docker-compose.yml)
#  API_KEY            — если задан, нужен заголовок Authorization: Bearer <key>.
#  MODEL              — HF id модели (ESpeech/ESpeech-TTS-1_RL-V2).
#  DEVICE             — auto (по умолчанию) | cuda | cpu.
#  VOICES_DIR         — каталог эталонов (.wav [+ .txt транскрипт]).
#  DEFAULT_SPEAKER    — голос по умолчанию; пусто → первый доступный.
#  USE_RUACCENT       — "1": ударения в /accent и (в plus-режиме) в синтезе.
#  ESPEECH_NFE        — шаги ODE-солвера (16–64; больше = лучше и медленнее).
#  ESPEECH_CFG        — guidance (сила следования тексту).
#  ESPEECH_SPEED      — базовая скорость речи модели.
#  ESPEECH_SYNTH_STRESS — plus (по умолчанию, модель обучена на «+») |
#                       mark | strip.
#  MAX_TEXT_CHARS     — предохранитель от гигантских запросов.
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("API_KEY", "").strip()
MODEL = os.environ.get("MODEL", "ESpeech/ESpeech-TTS-1_RL-V2").strip()
CACHE = os.environ.get("HF_HOME", "/opt/hf-cache").strip()
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices").strip()
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "").strip()
USE_RUACCENT = os.environ.get("USE_RUACCENT", "1").strip().lower() in ("1", "true", "yes")
NFE = max(8, min(64, int(os.environ.get("ESPEECH_NFE", "32"))))
CFG = float(os.environ.get("ESPEECH_CFG", "2.0"))
BASE_SPEED = float(os.environ.get("ESPEECH_SPEED", "1.0"))
STRESS_MODE = os.environ.get("ESPEECH_STRESS_MODE", "mark").strip().lower()
# ВАЖНО: модель обучена на «+» перед ударной гласной — plus по умолчанию.
SYNTH_STRESS = os.environ.get("ESPEECH_SYNTH_STRESS", "plus").strip().lower()
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))
SENT_PAUSE_MS = float(os.environ.get("ESPEECH_SENT_PAUSE_MS", "200"))
# F5 стабильнее, когда кусок сопоставим с ДЛИНОЙ ТРАНСКРИПТА РЕФЕРЕНСА
# (12 с ≈ 180 символов): на кусках заметно длиннее модель изредка глотает
# слова в середине («н+е б+ыло» пропадало на чанках 250+).
CHUNK_CHARS = int(os.environ.get("ESPEECH_CHUNK_CHARS", "160"))
REF_MAX_SEC = float(os.environ.get("ESPEECH_REF_MAX_SEC", "12"))
# Переопределение файлов чекпойнта/словаря, если авто-поиск не справился.
CKPT_OVERRIDE = os.environ.get("ESPEECH_CKPT", "").strip()
VOCAB_OVERRIDE = os.environ.get("ESPEECH_VOCAB", "").strip()

os.makedirs(VOICES_DIR, exist_ok=True)

app = FastAPI(title="ESpeech (F5-TTS ru) Server")

_dev_env = os.environ.get("DEVICE", "auto").strip().lower()
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _dev_env == "auto" else _dev_env
print("=" * 60)
print(f"Loading ESpeech ({MODEL}) on {DEVICE}...")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Файлы модели: чекпойнт + vocab из HF-снапшота (запечён в образ) ──────────
from huggingface_hub import snapshot_download

_snap = snapshot_download(repo_id=MODEL)


def _find_first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(_snap, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None


CKPT = CKPT_OVERRIDE or _find_first(["*.safetensors", "*.pt", "*.ckpt"])
VOCAB = VOCAB_OVERRIDE or _find_first(["vocab*.txt", "*.vocab", "*vocab*"])
if not CKPT:
    raise RuntimeError(f"Не найден чекпойнт в {_snap}; задайте ESPEECH_CKPT")
print(f"ckpt: {CKPT}\nvocab: {VOCAB}")

# ── F5-TTS высокоуровневый API ───────────────────────────────────────────────
from f5_tts.api import F5TTS

tts = F5TTS(ckpt_file=CKPT, vocab_file=VOCAB or "", device=DEVICE)
SAMPLE_RATE = 24000   # F5/Vocos — 24 кГц

# ── Ударения (RUAccent) — тот же контракт /accent, что у остальных ───────────
_STRESS = "́"
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
    """Текст для МОДЕЛИ. plus: ударения в формате «+» — ESpeech обучен именно
    на нём. Если клиент уже прислал метки (U+0301 из словаря приложения или
    «+») — просто конвертируем, RUAccent не дёргаем."""
    if SYNTH_STRESS == "plus":
        if _STRESS in text or "+" in text:
            return _MARKED_VOWEL.sub(r"+\1", _plus_to_mark(text))
        if _accentizer is not None:
            try:
                marked = _accentizer.process_all(text)
                return _MARKED_VOWEL.sub(r"+\1", marked)
            except Exception as e:
                print(f"RUAccent synth error: {e}")
    return _apply_stress_format(text, SYNTH_STRESS)


# ── Голоса (эталоны для клонирования — как у XTTS/VoxCPM) ────────────────────
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
    """Путь к эталону: запрошенный → DEFAULT_SPEAKER → первый доступный.
    None — только когда каталог пуст (для F5 это фатально: референс обязателен)."""
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


# ── Транскрипт референса: <voice>.txt рядом с wav, иначе faster-whisper ─────
_whisper = None
_whisper_lock = threading.Lock()


def _get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            # CPU/int8: транскрипция разовая (кешируется в .txt), VRAM не трогаем.
            _whisper = WhisperModel("small", device="cpu", compute_type="int8")
        return _whisper


def ref_text_for(path: str) -> str:
    """Транскрипт РОВНО ТОГО файла, что уйдёт в модель. КРИТИЧНО: F5 оценивает
    темп речи по отношению длины ref_text к длительности ref-аудио. Транскрипт
    полного файла при обрезанном аудио = сверхбыстрое чтение и мусорные
    «хвосты» референса на стыках предложений (es-1 болел именно этим)."""
    txt = os.path.splitext(path)[0] + ".txt"
    if os.path.isfile(txt):
        try:
            t = open(txt, encoding="utf-8").read().strip()
            if t:
                return _tidy_ref_text(t)
        except Exception:
            pass
    print(f"Transcribing reference (once): {os.path.basename(path)}")
    segments, _info = _get_whisper().transcribe(path, language="ru", beam_size=1)
    text = " ".join(s.text.strip() for s in segments).strip()
    text = _tidy_ref_text(text)
    try:
        with open(txt, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"ref txt cache write failed: {e}")
    return text


def _tidy_ref_text(t: str) -> str:
    """Финальная пунктуация обязательна: без неё F5 «склеивает» конец
    референса с началом текста — слышны последние буквы референса.
    Многоточия нормализуются так же, как в тексте синтеза (whisper любит
    обрывать транскрипт обрезанного куска на «...»)."""
    t = _tame_for_synth(t.strip())
    if t and t[-1] not in ".!?":
        t += "."
    return t + " "


# Кэш обрезанных референсов (моно 24 кГц, ≤ REF_MAX_SEC) — В КАТАЛОГЕ ГОЛОСОВ,
# чтобы вместе с парным .txt переживать перезапуск контейнера (в /tmp кеш
# стирался и транскрипция повторялась на каждом старте).
_REF_CACHE_DIR = os.path.join(VOICES_DIR, ".refcache")
os.makedirs(_REF_CACHE_DIR, exist_ok=True)


def _probe_duration(path: str) -> Optional[float]:
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def _trimmed_ref(path: str) -> str:
    """Короткий эталон (≤ REF_MAX_SEC) отдаётся как есть — тогда работает и
    пользовательский <voice>.txt. Длинный режется ffmpeg'ом, и транскрипт
    берётся уже от ОБРЕЗАННОГО куска (см. ref_text_for)."""
    import subprocess
    try:
        dur = _probe_duration(path)
        if dur is not None and dur <= REF_MAX_SEC + 0.5:
            return path
        st = os.stat(path)
        key = f"{os.path.basename(path)}_{int(st.st_mtime)}_{int(REF_MAX_SEC)}.wav"
        out = os.path.join(_REF_CACHE_DIR, key)
        if os.path.isfile(out):
            return out
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-t", str(REF_MAX_SEC), "-ac", "1", "-ar", "24000", out],
            check=True, timeout=60,
        )
        return out
    except Exception as e:
        print(f"ref trim failed ({e}); using original")
        return path


# ── Синтез ────────────────────────────────────────────────────────────────────
_HAS_SPEECH = re.compile(r"[^\W_]", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# Многоточия ломают выравнивание F5: после «... —» (обычная связка в русских
# диалогах) модель ГЛОТАЕТ следующие слова («— На помощь... — заорал один» →
# «заорал один» пропадало). Приводим пунктуацию к формам, на которых модель
# стабильна: «?..»/«!..» → «?»/«!», хвостовое многоточие → точка, многоточие
# в середине → запятая (пауза сохраняется, слова не теряются).
_ELL_AFTER_BANG = re.compile(r"([!?])\.{1,}")
_ELL_TAIL = re.compile(r"\.{2,}\s*$")
_ELL_MID = re.compile(r"\.{2,}")
_DUP_COMMA = re.compile(r",\s*,+")


def _tame_for_synth(t: str) -> str:
    t = t.replace("…", "...")
    t = _ELL_AFTER_BANG.sub(r"\1", t)
    t = _ELL_TAIL.sub(".", t)
    t = _ELL_MID.sub(",", t)
    t = _DUP_COMMA.sub(",", t)
    return t


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
    return out


# Приложение предрендерит абзацы параллельно — генерация строго по одной.
_gen_lock = threading.Lock()


def _generate_one(text: str, ref_path: str, ref_text: str) -> np.ndarray:
    with _gen_lock:
        try:
            wav, sr, _ = tts.infer(
                ref_file=ref_path,
                ref_text=ref_text,
                gen_text=text,
                nfe_step=NFE,
                cfg_strength=CFG,
                speed=BASE_SPEED,
                remove_silence=False,
            )
        finally:
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    return np.asarray(wav, dtype=np.float32).flatten()


def synthesize(text: str, voice: Optional[str]) -> io.BytesIO:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    text = _tame_for_synth(_stress_for_synth(text))

    speaker = resolve_speaker(voice)
    if speaker is None:
        raise HTTPException(
            status_code=503,
            detail="Нет ни одного эталона голоса. F5/ESpeech требует референс: "
                   "загрузите wav через /voices/upload (6–15 с чистой речи).",
        )
    # Аудио и транскрипт — ОДИН И ТОТ ЖЕ файл (обрезанный или короткий
    # оригинал). Раньше сюда попадал транскрипт полного файла при обрезанном
    # аудио — модель читала со скоростью «весь текст за 12 секунд».
    ref = _trimmed_ref(speaker)
    rtext = ref_text_for(ref)

    chunks = _chunks(text)
    if not chunks:
        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32), SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return buf

    pause = np.zeros(int(SAMPLE_RATE * SENT_PAUSE_MS / 1000.0), dtype=np.float32)
    audio = _generate_one(chunks[0], ref, rtext)
    for c in chunks[1:]:
        audio = np.concatenate([audio, pause, _generate_one(c, ref, rtext)])

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf


# ── HTTP-контракт (идентичен XTTS/Supertonic/VoxCPM) ─────────────────────────

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
    # Транскрипт сразу (CPU, разово) — чтобы первый /tts не ждал whisper.
    # Греем именно ОБРЕЗАННЫЙ вариант: он и уйдёт в модель.
    try:
        ref_text_for(_trimmed_ref(dest))
    except Exception as e:
        print(f"upload transcribe failed (отложится до первого /tts): {e}")
    return {"ok": True, "voice": slug}


@app.delete("/voices/{name}")
async def delete_voice(name: str, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    slug = safe_voice_name(name)
    deleted = False
    for ext in _AUDIO_EXTS + (".txt",):
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
        "engine": "espeech-f5",
        "model": MODEL,
        "server_version": SERVER_VERSION,
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
        "nfe": NFE,
        "cfg": CFG,
        "synth_stress": SYNTH_STRESS,
        "ruaccent": _accentizer is not None,
        "voices": list_voice_names(),
    }


print("=" * 60)
print(f"  Reciter ESpeech server — SERVER_VERSION {SERVER_VERSION}")
print("=" * 60)
print(f"Device: {DEVICE}; nfe={NFE}, cfg={CFG}, synth_stress={SYNTH_STRESS}")
print(f"API auth: {'ON' if API_KEY else 'OFF (LAN only!)'}")
print(f"Voices dir: {VOICES_DIR} -> {list_voice_names()}")
print(f"RUAccent: {'ON' if _accentizer is not None else 'OFF'}")
