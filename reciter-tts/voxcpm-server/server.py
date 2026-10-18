import io
import os
import re
import tempfile

import numpy as np
import soundfile as sf

# Метка версии server.py — печатается при старте и отдаётся в /health.
SERVER_VERSION = "vc-8 (2026-07-12: VoxCPM2 на 8 ГБ для русского)"

from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

import torch
from voxcpm import VoxCPM

# ─────────────────────────────────────────────────────────────────────────────
#  VoxCPM2 сервер для Reciter — тот же API, что у XTTS/Supertonic серверов
#  (/tts, /accent, /voices, /health): приложение переключается сменой адреса
#  (порт 8004 в compose). Клонирование по wav-образцу ЕСТЬ (как у XTTS).
#
#  Конфигурация через окружение (docker-compose.yml)
#  API_KEY            — если задан, нужен заголовок Authorization: Bearer <key>.
#  MODEL              — HF id модели (openbmb/VoxCPM2). Качается при первом
#                       старте в /cache (том), не в образ.
#  DEVICE             — auto (по умолчанию) | cuda | cpu. auto: cuda при
#                       наличии, иначе CPU (медленнее, но работает).
#  VOICES_DIR         — каталог эталонов голоса (.wav) для клонирования.
#  DEFAULT_SPEAKER    — голос по умолчанию; пусто/нет файла → встроенный
#                       голос модели (генерация без референса).
#  USE_RUACCENT       — "1": ударения в /accent (серверный режим приложения).
#  VOXCPM_CFG         — guidance (сила следования тексту/референсу).
#  VOXCPM_TIMESTEPS   — шаги диффузии CFM (больше = лучше и медленнее).
#  VOXCPM_SYNTH_STRESS— что скармливать модели: strip (по умолчанию,
#                       поддержка меток не документирована) | mark | plus.
#  MAX_TEXT_CHARS     — предохранитель от гигантских запросов.
# ─────────────────────────────────────────────────────────────────────────────
import threading

API_KEY = os.environ.get("API_KEY", "").strip()
REQUESTED_MODEL = os.environ.get("MODEL", "").strip()
# VoxCPM-0.5B — быстрый fallback, но он обучен на китайском/английском и для
# русского звучит с китайским акцентом. Поэтому явный VOXCPM_MODEL=VoxCPM2
# уважаем на 8-ГБ картах; ниже порога оставляем защитный откат.
LOW_VRAM_MODEL = os.environ.get("VOXCPM_LOW_VRAM_MODEL", "openbmb/VoxCPM-0.5B").strip()
FULL_MODEL = os.environ.get("VOXCPM_FULL_MODEL", "openbmb/VoxCPM2").strip()
FULL_MIN_VRAM_GB = float(os.environ.get("VOXCPM_FULL_MIN_VRAM_GB", "7"))
ALLOW_FULL_ON_LOW_VRAM = os.environ.get("VOXCPM_ALLOW_FULL_ON_LOW_VRAM", "0").strip().lower() in ("1", "true", "yes")
# По умолчанию — каталог с ЗАПЕЧЁННЫМИ в образ весами (см. Dockerfile,
# PREFETCH_MODEL=1). Переопределяется через HF_HOME (например на том).
CACHE = os.environ.get("HF_HOME", "/opt/hf-cache").strip()
# Запечённый образ должен стартовать без DNS/интернета. Для тонкого образа с
# первым скачиванием в том поставьте VOXCPM_OFFLINE=0 в .env/compose.
if os.environ.get("VOXCPM_OFFLINE", "1").strip().lower() not in ("0", "false", "no"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
MODEL = REQUESTED_MODEL or LOW_VRAM_MODEL
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices").strip()
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "").strip()
USE_RUACCENT = os.environ.get("USE_RUACCENT", "1").strip().lower() in ("1", "true", "yes")
CFG = float(os.environ.get("VOXCPM_CFG", "2.0"))
TIMESTEPS = int(os.environ.get("VOXCPM_TIMESTEPS", "10"))
# Референс длиннее этого режется (ffmpeg): длинный эталон при VAE-кодировании
# не влезает в память 8-ГБ карты (OOM в build_prompt_cache) и не нужен —
# VoxCPM хватает ~10 с чистой речи.
REF_MAX_SEC = float(os.environ.get("VOXCPM_REF_MAX_SEC", "12"))
# Фиксированный seed: без референса VoxCPM сэмплирует СЛУЧАЙНЫЙ голос на
# каждый вызов — с фиксированным seed «встроенный» голос стабилен.
SEED = int(os.environ.get("VOXCPM_SEED", "42"))
# Экономия VRAM: денойзер и CUDA-графы (optimize) можно выключить.
LOAD_DENOISER = os.environ.get("VOXCPM_DENOISER", "0").strip().lower() in ("1", "true", "yes")
OPTIMIZE = os.environ.get("VOXCPM_OPTIMIZE", "0").strip().lower() in ("1", "true", "yes")
STRESS_MODE = os.environ.get("VOXCPM_STRESS_MODE", "mark").strip().lower()
SYNTH_STRESS = os.environ.get("VOXCPM_SYNTH_STRESS", "strip").strip().lower()
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))
SENT_PAUSE_MS = float(os.environ.get("VOXCPM_SENT_PAUSE_MS", "200"))
# Абзацы длиннее этого делятся по предложениям (стабильность длинной генерации).
CHUNK_CHARS = int(os.environ.get("VOXCPM_CHUNK_CHARS", "400"))

os.makedirs(VOICES_DIR, exist_ok=True)

app = FastAPI(title="VoxCPM2 Server")

# ── Устройство: auto → cuda при наличии, иначе CPU ───────────────────────────
_dev_env = os.environ.get("DEVICE", "auto").strip().lower()
if _dev_env == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = _dev_env
print("=" * 60)
GPU_VRAM_GB = None
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    GPU_VRAM_GB = props.total_memory / (1024 ** 3)
    if REQUESTED_MODEL == FULL_MODEL and GPU_VRAM_GB < FULL_MIN_VRAM_GB and not ALLOW_FULL_ON_LOW_VRAM:
        print(
            f"VOXCPM_MODEL={FULL_MODEL} ignored: GPU has {GPU_VRAM_GB:.1f} GB, "
            f"minimum guard is {FULL_MIN_VRAM_GB:.1f} GB. Using {LOW_VRAM_MODEL}. "
            "Russian synthesis needs VoxCPM2; lower VOXCPM_FULL_MIN_VRAM_GB "
            "only if you accept OOM risk."
        )
        MODEL = LOW_VRAM_MODEL
print(f"Loading VoxCPM ({MODEL}) on {DEVICE}...")
if DEVICE == "cuda":
    print(f"GPU: {props.name}; VRAM: {GPU_VRAM_GB:.1f} GB")
    if REQUESTED_MODEL == "" and GPU_VRAM_GB >= FULL_MIN_VRAM_GB and LOW_VRAM_MODEL != FULL_MODEL:
        print(
            f"Low-VRAM default is active: {MODEL}. "
            f"Set VOXCPM_MODEL={FULL_MODEL} and rebuild the image to use Russian/multilingual synthesis."
        )

# from_pretrained: сигнатура меняется между версиями пакета — необязательные
# kwargs (device/cache_dir/optimize) отбрасываются по одному при TypeError,
# чтобы сервер поднимался на любой 2.x.
def _load_model():
    kwargs = dict(cache_dir=CACHE, device=DEVICE, optimize=OPTIMIZE, load_denoiser=LOAD_DENOISER)
    drop_order = ["optimize", "device", "cache_dir", "load_denoiser"]
    while True:
        try:
            return VoxCPM.from_pretrained(MODEL, **kwargs)
        except TypeError as e:
            dropped = False
            for k in drop_order:
                if k in kwargs and k in str(e):
                    kwargs.pop(k); dropped = True; break
            if not dropped:
                if kwargs:
                    kwargs.pop(next(iter(kwargs))); continue
                raise

tts = _load_model()

# Частота дискретизации — ИЗ МОДЕЛИ (VoxCPM2 = 48 кГц). Хардкод (напр. 24000)
# делает голос вдвое ниже и медленнее.
SAMPLE_RATE = int(
    getattr(getattr(tts, "tts_model", None), "sample_rate", 0)
    or os.environ.get("VOXCPM_SAMPLE_RATE", "48000")
)
print(f"Sample rate: {SAMPLE_RATE}")

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
    """«+»-метки (перед гласной) → U+0301 после гласной; прочие «+» не трогаются."""
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


# ── Голоса (эталоны для клонирования, как у XTTS) ────────────────────────────
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_\-]+")


def safe_voice_name(name: str) -> str:
    slug = _SAFE_NAME.sub("_", (name or "").strip())[:64]
    return slug or "voice"


def list_voice_names():
    try:
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(VOICES_DIR)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
        )
    except FileNotFoundError:
        return []


def resolve_speaker(voice: Optional[str]) -> Optional[str]:
    """Путь к эталону: запрошенный → DEFAULT_SPEAKER → ПЕРВЫЙ доступный файл.
    Откат на первый файл важен: без референса VoxCPM сэмплирует случайный
    голос на каждый запрос («каждый раз разными голосами»). None — только
    когда каталог голосов пуст."""
    for name in (voice, DEFAULT_SPEAKER):
        if not name:
            continue
        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            p = os.path.join(VOICES_DIR, f"{safe_voice_name(name)}{ext}")
            if os.path.isfile(p):
                return p
    names = list_voice_names()
    if names:
        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            p = os.path.join(VOICES_DIR, f"{names[0]}{ext}")
            if os.path.isfile(p):
                return p
    return None


# Кэш обрезанных референсов: длинный эталон один раз режется ffmpeg'ом до
# REF_MAX_SEC (и приводится к моно) — иначе VAE-кодирование референса даёт
# CUDA OOM на 8-ГБ картах. Ключ содержит mtime — перезаписанный голос
# подхватится сам.
_REF_CACHE_DIR = os.path.join(tempfile.gettempdir(), "voxcpm_refcache")
os.makedirs(_REF_CACHE_DIR, exist_ok=True)


def _trimmed_ref(path: str) -> str:
    import subprocess
    try:
        mtime = int(os.path.getmtime(path))
    except OSError:
        return path
    dest = os.path.join(
        _REF_CACHE_DIR,
        f"{safe_voice_name(os.path.splitext(os.path.basename(path))[0])}_{mtime}_{int(REF_MAX_SEC)}.wav",
    )
    if os.path.isfile(dest):
        return dest
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-t", str(REF_MAX_SEC), "-ac", "1", dest],
        capture_output=True,
    )
    if r.returncode != 0 or not os.path.isfile(dest):
        print(f"ref trim failed for {path}: {r.stderr.decode(errors='ignore')[-200:]}")
        return path
    return dest


print("=" * 60)
print(f"  Reciter VoxCPM server — SERVER_VERSION {SERVER_VERSION}")
print("=" * 60)
print(f"Device: {DEVICE}; cfg={CFG}, timesteps={TIMESTEPS}")
print(f"API auth: {'ON' if API_KEY else 'OFF (LAN only!)'}")
print(f"Voices dir: {VOICES_DIR} -> {list_voice_names() or 'EMPTY (встроенный голос модели)'}")
print(f"RUAccent: {'ON' if _accentizer else 'OFF'}")


def supports_reference_wav() -> bool:
    """VoxCPM-0.5B не принимает reference_wav_path; клонирование доступно только в полной VoxCPM2."""
    return MODEL == FULL_MODEL or "voxcpm2" in MODEL.lower()


def supports_russian() -> bool:
    """Русский нужен полной VoxCPM2; 0.5B — китайско-английская модель."""
    return supports_reference_wav()


if not supports_reference_wav() and list_voice_names():
    print(
        f"Voice references are ignored for {MODEL}: reference_wav_path is supported only by VoxCPM2. "
        "Russian text also needs VoxCPM2; VoxCPM-0.5B is Chinese/English and may sound Chinese."
    )


def require_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    import hmac
    expected = f"Bearer {API_KEY}"
    if not authorization or not hmac.compare_digest(authorization.strip(), expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


# «Речевой» символ — буква/цифра. Куски без них (одни знаки препинания) в
# модель не идут: на голой пунктуации мультиязычные TTS галлюцинируют
# (озвучивают знаки на случайном языке).
_HAS_SPEECH = re.compile(r"[^\W_]", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _chunks(text: str):
    """Предложения, сгруппированные в куски ≤ CHUNK_CHARS — стабильная длина
    генерации для диффузионной модели. Куски без речи отбрасываются."""
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


# GPU-замок: приложение предрендерит абзацы ПАРАЛЛЕЛЬНО (2-3 запроса разом),
# а две одновременные генерации VoxCPM не влезают в 8 ГБ VRAM → OOM.
# Генерация строго последовательная; параллельные запросы ждут очереди.
_gen_lock = threading.Lock()


def _generate_one(text: str, speaker: Optional[str]) -> np.ndarray:
    kwargs = dict(text=text, cfg_value=CFG, inference_timesteps=TIMESTEPS)
    if speaker and supports_reference_wav():
        kwargs["reference_wav_path"] = _trimmed_ref(speaker)
    with _gen_lock:
        try:
            # Детерминизм без kwarg: voxcpm.generate() не принимает seed.
            try:
                torch.manual_seed(SEED)
                if DEVICE == "cuda":
                    torch.cuda.manual_seed_all(SEED)
            except Exception:
                pass
            wav = tts.generate(**kwargs)
        finally:
            # Освобождаем кэш VRAM между запросами — на 8-ГБ карте
            # фрагментация иначе копится и приводит к OOM на 3-4-й генерации.
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    return np.asarray(wav, dtype=np.float32).flatten()


def synthesize(text: str, voice: Optional[str]) -> io.BytesIO:
    text = _apply_stress_format((text or "").strip(), SYNTH_STRESS)
    if not text:
        raise ValueError("Empty text")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    chunks = _chunks(text)
    if not chunks:
        # Весь вход — пунктуация/пробелы: короткая тишина вместо мусора.
        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32), SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return buf

    speaker = resolve_speaker(voice)
    pause = np.zeros(int(SAMPLE_RATE * SENT_PAUSE_MS / 1000.0), dtype=np.float32)
    audio = _generate_one(chunks[0], speaker)
    for c in chunks[1:]:
        audio = np.concatenate([audio, pause, _generate_one(c, speaker)])

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf


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
        return StreamingResponse(synthesize(request.input, request.voice), media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    language: str = Form("ru"),          # VoxCPM определяет язык по тексту — параметр принимается для совместимости
    voice: Optional[str] = Form(None),
    accent: Optional[bool] = Form(False),
    speaker_wav: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(None),
):
    require_auth(authorization)
    if accent:
        text = accentize(text)
    tmp_upload = None
    try:
        if speaker_wav:
            suffix = os.path.splitext(speaker_wav.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await speaker_wav.read())
                tmp_upload = tmp.name
            if supports_reference_wav():
                # Прямо переданный референс важнее сохранённого имени.
                buf = io.BytesIO()
                t = _apply_stress_format((text or "").strip(), SYNTH_STRESS)[:MAX_TEXT_CHARS]
                audio = _generate_one(t, tmp_upload)
                sf.write(buf, audio, SAMPLE_RATE, format="WAV")
                buf.seek(0)
                return StreamingResponse(buf, media_type="audio/wav")
            print(f"Uploaded speaker_wav is ignored for {MODEL}: reference_wav_path is supported only by VoxCPM2.")
        return StreamingResponse(synthesize(text, voice), media_type="audio/wav")
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_upload and os.path.exists(tmp_upload):
            os.unlink(tmp_upload)


@app.get("/voices")
async def list_voices(authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    return {"voices": list_voice_names(), "default": DEFAULT_SPEAKER, "reference_wav": supports_reference_wav()}


@app.post("/voices/upload")
async def upload_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """Эталон голоса: 6–20 сек чистой речи. Перекодируется в WAV 48 кГц моно
    через ffmpeg — надёжно для любого присланного формата."""
    require_auth(authorization)
    import subprocess
    dest = os.path.join(VOICES_DIR, f"{safe_voice_name(name)}.wav")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or "")[1] or ".bin", delete=False) as tmp:
        tmp.write(data)
        tmp_in = tmp.name
    try:
        cmd = ["ffmpeg", "-y", "-i", tmp_in, "-ar", str(SAMPLE_RATE), "-ac", "1", dest]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not os.path.isfile(dest):
            tail = r.stderr.decode(errors="ignore")[-300:]
            return JSONResponse(status_code=400, content={"error": f"decode failed: {tail}"})
        return {"ok": True, "name": safe_voice_name(name), "voices": list_voice_names()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Bad audio: {e}"})
    finally:
        if os.path.exists(tmp_in):
            os.unlink(tmp_in)


@app.delete("/voices/{name}")
async def delete_voice(name: str, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    removed = False
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        p = os.path.join(VOICES_DIR, f"{safe_voice_name(name)}{ext}")
        if os.path.isfile(p):
            os.unlink(p)
            removed = True
    return {"ok": removed, "voices": list_voice_names()}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "server_version": SERVER_VERSION,
        "engine": "voxcpm",
        "model": MODEL,
        "requested_model": REQUESTED_MODEL,
        "supports_russian": supports_russian(),
        "reference_wav": supports_reference_wav(),
        "low_vram_model": LOW_VRAM_MODEL,
        "full_model": FULL_MODEL,
        "full_min_vram_gb": FULL_MIN_VRAM_GB,
        "allow_full_on_low_vram": ALLOW_FULL_ON_LOW_VRAM,
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
        "auth": bool(API_KEY),
        "ruaccent": bool(_accentizer),
        "accent_response_mode": STRESS_MODE,
        "synth_stress": SYNTH_STRESS,
        "cfg": CFG,
        "timesteps": TIMESTEPS,
        "seed": SEED,
        "ref_max_sec": REF_MAX_SEC,
        "denoiser": LOAD_DENOISER,
        "optimize": OPTIMIZE,
        "voices": list_voice_names(),
        "default_voice": DEFAULT_SPEAKER,
        "gpu_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "N/A",
    }
