import io
import os
import re
import tempfile
import torch
import numpy as np
import soundfile as sf

# Метка версии server.py. При каждом изменении файла увеличивается — печатается
# при старте и отдаётся в /health, чтобы за секунду проверить, что в контейнере
# крутится СВЕЖИЙ файл, а не старая копия. Если в docker-логе версия ниже —
# обнови server.py на ПК и пересобери (docker compose up -d --build).
SERVER_VERSION = "10 (2026-07-18: своё деление длинных предложений + «стоп» после тире/кавычек — меньше хвостовых артефактов)"

# --- ПАТЧ 1: PYTORCH 2.6+ (weights_only=False) ---
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

# --- ПАТЧ 2: TORCHAUDIO (обход ошибки TorchCodec через soundfile) ---
import torchaudio
def _patched_ta_load(filepath, *args, **kwargs):
    audio_np, sr = sf.read(filepath, always_2d=True)
    audio_tensor = torch.from_numpy(audio_np).float().T
    return audio_tensor, sr
torchaudio.load = _patched_ta_load

from fastapi import FastAPI, Form, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import TTS as _tts_pkg
from TTS.api import TTS

# Версия установленной библиотеки Coqui TTS. Форк coqui-tts (idiap) — 0.24+;
# оригинальный заброшенный пакет tts застрял на 0.22.0. Печатается при старте
# и отдаётся в /health, чтобы проверить, что в образе именно поддерживаемый форк.
TTS_LIB_VERSION = getattr(_tts_pkg, "__version__", "?")

# ─────────────────────────────────────────────────────────────────────────────
#  Конфигурация через окружение (docker-compose.yml)
# ─────────────────────────────────────────────────────────────────────────────
#  API_KEY             — если задан, запросы обязаны прислать заголовок
#                        Authorization: Bearer <key>. Пусто = без авторизации
#                        (ТОЛЬКО для изолированной локальной сети!).
#  VOICES_DIR          — каталог с эталонными голосами (.wav) для клонирования.
#                        Приложение загружает/выбирает/удаляет их через /voices.
#  DEFAULT_SPEAKER     — имя голоса из VOICES_DIR, используемого по умолчанию.
#  USE_RUACCENT        — "1"/"true": проставлять ударения (RUAccent) в /accent и,
#                        опционально, в /tts (см. параметр accent).
#  MAX_TEXT_CHARS      — предохранитель от гигантских запросов.
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("API_KEY", "").strip()
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/voices").strip()
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "default").strip()
USE_RUACCENT = os.environ.get("USE_RUACCENT", "1").strip().lower() in ("1", "true", "yes")
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))

# ── Борьба с «мусорным хвостом» XTTS ──────────────────────────────────────────
# XTTS-v2 склонна доклеивать в конце фразы короткий посторонний звук (призвук,
# «бормотание», щелчок) — особенно на коротких репликах, когда GPT-декодер не
# получил чёткого сигнала «стоп». Лечим тремя независимыми способами:
#   1) Гарантируем финальную пунктуацию (стоп-сигнал для модели) — _ensure_stop.
#   2) Параметры сэмплинга против «залипания»/добора токенов — _xtts_gen_kwargs.
#      Если хвост всё равно слышен — поднимите XTTS_REPETITION_PENALTY (8–12).
#   3) Пост-обработка волны: срез хвостовой тишины и короткого «выброса» после
#      явной паузы + мягкий fade — _postprocess.
def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _envi(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except ValueError:
        return int(default)


XTTS_TEMPERATURE = _envf("XTTS_TEMPERATURE", 0.7)
XTTS_LENGTH_PENALTY = _envf("XTTS_LENGTH_PENALTY", 1.0)
XTTS_REPETITION_PENALTY = _envf("XTTS_REPETITION_PENALTY", 6.0)
XTTS_TOP_K = _envi("XTTS_TOP_K", 50)
XTTS_TOP_P = _envf("XTTS_TOP_P", 0.85)
# Пост-обработка: включена по умолчанию, всё настраивается из окружения.
TRIM_SILENCE = os.environ.get("XTTS_TRIM_SILENCE", "1").strip().lower() in ("1", "true", "yes")
TRIM_TAIL = os.environ.get("XTTS_TRIM_TAIL", "1").strip().lower() in ("1", "true", "yes")
SILENCE_DB = _envf("XTTS_SILENCE_DB", -45.0)      # порог тишины относительно пика
TAIL_MIN_GAP_MS = _envf("XTTS_TAIL_MIN_GAP_MS", 200.0)   # пауза перед «выбросом»
TAIL_MAX_LEN_MS = _envf("XTTS_TAIL_MAX_LEN_MS", 250.0)   # макс. длина «выброса»
FADE_MS = _envf("XTTS_FADE_MS", 12.0)

os.makedirs(VOICES_DIR, exist_ok=True)

app = FastAPI(title="XTTS-v2 Server (GPU)")

print("Loading XTTS-v2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Модель качается при ПЕРВОМ старте в TTS_HOME (том /cache/tts в compose) и
# переживает пересборки образа. Согласие с лицензией CPML — файл .coqui_tos
# в каталоге модели (created here) + COQUI_TOS_AGREED=1 в окружении.
_tts_home = os.environ.get("TTS_HOME", "").strip()
if _tts_home:
    for _d in (_tts_home, os.path.join(_tts_home, "tts")):
        _mdir = os.path.join(_d, "tts_models--multilingual--multi-dataset--xtts_v2")
        os.makedirs(_mdir, exist_ok=True)
        open(os.path.join(_mdir, ".coqui_tos"), "a").close()

# XTTS_MODEL_DIR — каталог с КАСТОМНОЙ моделью XTTS (model.pth, config.json,
# vocab.json). Нужен для русских файнтюнов вроде xttsv2_banana, обученных на
# «+»-метках ударений (базовая XTTS-v2 ударений не понимает — см. STRESS ниже).
# Смонтируй каталог в compose и укажи путь; пусто = штатная XTTS-v2.
XTTS_MODEL_DIR = os.environ.get("XTTS_MODEL_DIR", "").strip()
if XTTS_MODEL_DIR:
    print(f"Custom model dir: {XTTS_MODEL_DIR}")
    tts = TTS(
        model_path=XTTS_MODEL_DIR,
        config_path=os.path.join(XTTS_MODEL_DIR, "config.json"),
        progress_bar=False,
    ).to(device)
else:
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False
    ).to(device)

try:
    speakers_list = tts.speakers if hasattr(tts, 'speakers') and tts.speakers else []
except Exception:
    speakers_list = []

SAMPLE_RATE = 24000

# ── RUAccent (ударения) ──────────────────────────────────────────────────────
# ФАКТЫ (проверено по коду Coqui TTS и опыту сообщества):
#   • Базовая XTTS-v2 ударений НЕ понимает. Её клинер для ru не удаляет
#     U+0301, а в BPE-словаре токенизатора этого символа нет → каждый акут
#     превращается в незнакомый токен, и модель ЗАПИНАЕТСЯ на нём. «+» в
#     тексте она тоже читает как мусор.
#   • Управляемые ударения в XTTS дают только русские файнтюны, обученные на
#     «+»-метках (напр. xttsv2_banana, формат «молок+о» — как у RUAccent).
# Поэтому форматы РАЗДЕЛЕНЫ:
#   XTTS_STRESS_MODE  — формат ответа /accent для ВНЕШНИХ движков
#                       (Edge/Google/системный понимают U+0301): mark|plus|strip.
#   XTTS_SYNTH_STRESS — что скармливать САМОЙ XTTS при синтезе:
#       strip (по умолчанию) — снять метки; базовая XTTS ставит ударения сама;
#       plus  — «+» перед гласной; ставь ВМЕСТЕ с XTTS_MODEL_DIR=<файнтюн>;
#       mark  — U+0301 (базовой XTTS противопоказано — запинки).
STRESS_MODE = os.environ.get("XTTS_STRESS_MODE", "mark").strip().lower()
SYNTH_STRESS = os.environ.get("XTTS_SYNTH_STRESS", "strip").strip().lower()
_STRESS = "́"                       # комбинируемый акут (ударение)
_RU_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
_MARKED_VOWEL = re.compile(f"([{''.join(sorted(_RU_VOWELS))}])" + _STRESS)

_accentizer = None
if USE_RUACCENT:
    try:
        from ruaccent import RUAccent
        _accentizer = RUAccent()
        # tiny/turbo — быстрые модели; big — точнее, но тяжелее. turbo3.1 —
        # хороший баланс для чтения книг.
        _accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True)
        print(f"RUAccent loaded (turbo3.1); /accent={STRESS_MODE}, synth={SYNTH_STRESS}.")
    except Exception as e:
        print(f"RUAccent NOT available ({e}); accents disabled.")
        _accentizer = None


def _plus_to_mark(text: str) -> str:
    """«+»-метки (RUAccent/Silero-стиль, «+» перед гласной) → U+0301 после
    гласной. «+» НЕ рядом с гласной (арифметика «2+2») сохраняется как есть.
    Текст без «+» не трогается."""
    if "+" not in text:
        return text
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "+":
            if i + 1 < n and text[i + 1] in _RU_VOWELS:
                out.append(text[i + 1]); out.append(_STRESS); i += 2; continue
            if out and out[-1] in _RU_VOWELS:   # защитно: «+» после гласной
                out.append(_STRESS); i += 1; continue
            out.append(ch); i += 1              # не метка ударения — оставляем
            continue
        out.append(ch); i += 1
    return "".join(out)


def _apply_stress_format(text: str, mode: str) -> str:
    """Приводит текст с ЛЮБОЙ разметкой ударений («+» и/или U+0301) к
    формату [mode]: mark → U+0301; plus → «+» перед гласной; strip → без меток."""
    t = _plus_to_mark(text)               # унифицируем в U+0301
    if mode == "strip":
        return t.replace(_STRESS, "")
    if mode == "plus":
        return _MARKED_VOWEL.sub(r"+\1", t)
    return t                              # mark


def accentize(text: str) -> str:
    """RUAccent + формат для ответа /accent (внешние движки)."""
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
    """Sanitises a voice name into a filesystem-safe slug."""
    slug = _SAFE_NAME.sub("_", (name or "").strip())[:64]
    return slug or "voice"


def voice_path(name: str) -> str:
    return os.path.join(VOICES_DIR, f"{safe_voice_name(name)}.wav")


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
    """Returns a path to the reference clip for [voice], falling back to the
    default, then to any available voice. None if the voices dir is empty."""
    candidates = []
    if voice:
        candidates.append(voice)
    if DEFAULT_SPEAKER:
        candidates.append(DEFAULT_SPEAKER)
    for name in candidates:
        # Принимаем либо точный файл, либо санитизированный .wav, который мы храним.
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


print("=" * 60)
print(f"  Reciter XTTS server — SERVER_VERSION {SERVER_VERSION}")
print("=" * 60)
print(f"Model loaded on {device}.")
print(f"Coqui TTS lib: {TTS_LIB_VERSION}")
print(f"API auth: {'ON' if API_KEY else 'OFF (LAN only!)'}")
print(f"Voices dir: {VOICES_DIR} -> {list_voice_names() or 'EMPTY'}")
print(f"RUAccent: {'ON' if _accentizer else 'OFF'}")


def require_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    import hmac
    expected = f"Bearer {API_KEY}"
    if not authorization or not hmac.compare_digest(authorization.strip(), expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


_STOP_CHARS = ".!?…"
# Закрывающие обёртки в конце реплики: точку-«стоп» ставим ПЕРЕД ними.
_TRAILING_WRAP = "»”\"'）)]"


def _ensure_stop(text: str) -> str:
    """Гарантирует финальный знак-«стоп». XTTS без терминальной пунктуации не
    получает сигнала окончания и доклеивает посторонний звук в конце фразы.
    Раньше точка добавлялась ТОЛЬКО после буквы/цифры — а реплики, кончающиеся
    на тире «—», двоеточие или закрывающую кавычку «»» (сплошь и рядом в
    русских диалогах), оставались без «стоп»-сигнала и давали хвостовой
    артефакт. Теперь точку ставим всегда, когда последний ЗНАЧИМЫЙ символ (без
    учёта закрывающих кавычек/скобок) не является терминатором."""
    t = (text or "").rstrip()
    if not t:
        return t
    j = len(t) - 1
    while j >= 0 and t[j] in _TRAILING_WRAP:
        j -= 1
    if j < 0:
        return t  # одни кавычки/скобки — не трогаем
    if t[j] in _STOP_CHARS:
        return t  # «стоп» уже есть (в т.ч. перед закрывающей кавычкой)
    # Тире/двоеточие/запятая или буква/цифра в конце — чёткого «стоп» нет.
    # Точку вставляем перед закрывающими кавычками, если они есть.
    return t[:j + 1] + "." + t[j + 1:]


def _xtts_gen_kwargs() -> dict:
    """Параметры сэмплинга XTTS против «добора»/«бормотания» в конце фразы.
    repetition_penalty обязан быть float — внутри XTTS на это стоит assert."""
    return dict(
        temperature=float(XTTS_TEMPERATURE),
        length_penalty=float(XTTS_LENGTH_PENALTY),
        repetition_penalty=float(XTTS_REPETITION_PENALTY),
        top_k=int(XTTS_TOP_K),
        top_p=float(XTTS_TOP_P),
    )


def _run_tts(kwargs: dict):
    """Зовёт модель с анти-артефактными параметрами; при несовместимой версии
    TTS (не принимает лишние kwargs) — откат к вызову с дефолтами."""
    try:
        return tts.tts(**kwargs, **_xtts_gen_kwargs())
    except TypeError as e:
        print(f"XTTS gen params unsupported ({e}); falling back to defaults.")
        return tts.tts(**kwargs)


def _normalize_stress_input(text: str) -> str:
    """Приводит ЛЮБОЙ входной текст /tts (с «+», U+0301 или без меток) к
    формату XTTS_SYNTH_STRESS. По умолчанию strip: базовая XTTS-v2 меток не
    понимает — U+0301 даёт незнакомые токены (запинки), «+» читается как
    мусор. Приложение шлёт сюда текст, уже прогнанный через /accent, — метки
    снимаются здесь, а не на клиенте, чтобы /accent оставался полезным для
    Edge/Google."""
    return _apply_stress_format(text, SYNTH_STRESS)


# ── Кэш эмбеддингов голоса ───────────────────────────────────────────────────
# tts.tts() пересчитывает conditioning-латенты референса на КАЖДЫЙ вызов —
# это лишние сотни мс на каждый абзац. Считаем один раз на файл голоса и
# переиспользуем; в ключе mtime, так что перезаписанный голос подхватится сам.
_latents_cache = {}


def _get_latents(speaker_wav: str):
    key = (speaker_wav, os.path.getmtime(speaker_wav))
    hit = _latents_cache.get(key)
    if hit is not None:
        return hit
    model = tts.synthesizer.tts_model
    gpt_cond, spk_emb = model.get_conditioning_latents(audio_path=[speaker_wav])
    if len(_latents_cache) > 8:          # голосов немного; кэшу расти незачем
        _latents_cache.clear()
    _latents_cache[key] = (gpt_cond, spk_emb)
    return gpt_cond, spk_emb


def _synth_one(text: str, speaker_wav: str, language: str):
    """Синтез ОДНОГО куска (предложение/клауза ≤ лимита) → np.float32. Основной
    путь — model.inference() с кэшированными латентами (быстрее и с полным
    контролем параметров); если внутренний API этой версии TTS иной — откат на
    публичный tts.tts().

    enable_text_splitting=False намеренно: длинные предложения мы дробим САМИ
    (см. _split_long) и чистим хвост КАЖДОГО куска через _postprocess. С True
    XTTS резал предложение внутри себя и склеивал подкуски БЕЗ очистки их
    хвостов — отсюда «артефакты после предложения»."""
    try:
        model = tts.synthesizer.tts_model
        gpt_cond, spk_emb = _get_latents(speaker_wav)
        out = model.inference(
            text, language, gpt_cond, spk_emb,
            enable_text_splitting=False,
            **_xtts_gen_kwargs(),
        )
        return np.asarray(out["wav"], dtype=np.float32).flatten()
    except Exception as e:
        print(f"low-level inference failed ({e}); falling back to tts.tts()")
        audio = _run_tts({"text": text, "speaker_wav": speaker_wav, "language": language})
        return np.asarray(audio, dtype=np.float32).flatten()


# Предложения делим САМИ (а не внутри XTTS): тогда анти-артефактная
# пост-обработка (_postprocess) чистит хвост КАЖДОГО предложения, а не только
# конец склеенного клипа — «странные звуки» между предложениями уходят тоже.
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
SENT_PAUSE_MS = _envf("XTTS_SENT_PAUSE_MS", 250.0)   # пауза между предложениями

# «Речевой» символ — любая буква или цифра (юникод). Куски БЕЗ них (одни знаки
# препинания: «—», «…», «»», кавычки, скобки, разделители «* * *») НЕЛЬЗЯ слать
# в XTTS: на «голой» пунктуации мультиязычная XTTS-v2 не определяет язык и
# ОЗВУЧИВАЕТ знаки на случайном языке — на слух похоже на китайский. Такие
# фрагменты просто пропускаем (вместо них — межпредложенческая пауза).
_HAS_SPEECH = re.compile(r"[^\W_]", re.UNICODE)


def _has_speech(text: str) -> bool:
    return bool(_HAS_SPEECH.search(text or ""))


# У XTTS жёсткий лимит ~182 символа на проход: предложение длиннее просто
# обрезается по звуку («The text length exceeds the character limit of 182»).
# Раньше это страховал внутренний сплиттер модели (enable_text_splitting=True),
# но он склеивал подкуски без очистки их хвостов. Теперь длинные предложения
# режем САМИ и синтезируем/чистим каждый кусок отдельно.
XTTS_MAX_CHARS = _envi("XTTS_MAX_CHARS", 180)
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—])\s+")


def _pack(parts, limit):
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > limit:
            out.append(cur); cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return out


def _split_long(s: str):
    """Предложение ≤ лимита — как есть. Длиннее — режем по клаузам
    (запятая/тире/двоеточие), а неделимо-длинную клаузу — по словам (крайняя
    мера, чтобы не превысить лимит XTTS)."""
    if len(s) <= XTTS_MAX_CHARS:
        return [s]
    clauses = [p for p in _CLAUSE_SPLIT.split(s) if p.strip()]
    packed = _pack(clauses, XTTS_MAX_CHARS) if len(clauses) > 1 else [s]
    out = []
    for c in packed:
        if len(c) <= XTTS_MAX_CHARS:
            out.append(c)
        else:
            out.extend(_pack(c.split(), XTTS_MAX_CHARS))
    return out


def _postprocess(audio, sr: int):
    """Убирает хвостовой «мусор» XTTS: (1) срез хвостовой/начальной тишины;
    (2) отрез короткого постороннего «выброса» в самом конце, если он отделён
    от речи явной паузой; (3) мягкий fade, чтобы срез не щёлкал."""
    a = np.asarray(audio, dtype=np.float32).flatten()
    n = a.size
    if n == 0:
        return a
    peak = float(np.max(np.abs(a))) or 1.0
    thr = peak * (10.0 ** (SILENCE_DB / 20.0))
    win = max(1, int(sr * 0.01))            # окна по 10 мс
    nwin = n // win
    if (TRIM_SILENCE or TRIM_TAIL) and nwin >= 3:
        frames = a[:nwin * win].reshape(nwin, win)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        idx = np.where(rms > thr)[0]
        if idx.size:
            first, last = int(idx[0]), int(idx[-1])
            # (2) Короткий «выброс» после явной паузы у самого конца — это почти
            # всегда галлюцинация модели, а не последний слог. Отрезаем его.
            if TRIM_TAIL and idx.size >= 2:
                min_gap = int(TAIL_MIN_GAP_MS / 10.0)   # окна 10 мс
                max_tail = int(TAIL_MAX_LEN_MS / 10.0)
                gaps = np.diff(idx)
                big = np.where(gaps >= min_gap)[0]
                if big.size:
                    g = int(big[-1])
                    body_end = int(idx[g])              # конец «тела» перед паузой
                    tail_start = int(idx[g + 1])
                    tail_len = last - tail_start + 1
                    body_len = body_end - first + 1
                    if tail_len <= max_tail and body_len >= tail_len:
                        last = body_end                 # выкидываем хвостовой выброс
            if not TRIM_SILENCE:
                first = 0                                # тишину не режем — только хвост
            pad = int(sr * 0.04)                         # 40 мс запаса, чтобы не «съесть» речь
            start = max(0, first * win - pad)
            end = min(n, (last + 1) * win + pad)
            a = a[start:end]
    fade = max(1, int(sr * FADE_MS / 1000.0))
    if a.size > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        a[:fade] *= ramp
        a[-fade:] *= ramp[::-1]
    return a


def synthesize(text: str, speaker_wav_path: str = None, language: str = "ru"):
    text = _normalize_stress_input((text or "").strip())
    if not text:
        raise ValueError("Empty text")
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    speaker = speaker_wav_path or resolve_speaker(None)
    if not speaker and not speakers_list:
        raise ValueError("No voice available: upload one via /voices/upload.")

    # Только предложения с речью (буква/цифра). Куски из одной пунктуации в
    # XTTS не идут — иначе модель озвучивает знаки на «китайском» (см. _has_speech).
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip() and _has_speech(s)]
    if not sentences:
        # Весь вход — пунктуация/пробелы (реплика-многоточие, строка-разделитель):
        # отдаём короткую тишину, а не синтезированный мусор.
        silence = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silence, SAMPLE_RATE, format='WAV')
        buf.seek(0)
        return buf
    if speaker:
        # По куску за проход (предложение или его клауза ≤ лимита XTTS): у
        # каждого чистится свой хвост, склейка через фиксированную паузу —
        # артефакты после предложений и внутри длинных предложений исчезают.
        pieces = []
        for s in sentences:
            pieces.extend(_split_long(s.strip()))
        chunks = []
        for p in pieces:
            a = _synth_one(_ensure_stop(p.strip()), speaker, language)
            chunks.append(_postprocess(a, SAMPLE_RATE))
        pause = np.zeros(int(SAMPLE_RATE * SENT_PAUSE_MS / 1000.0), dtype=np.float32)
        audio = chunks[0]
        for c in chunks[1:]:
            audio = np.concatenate([audio, pause, c])
    else:
        # Встроенный speaker без референс-файла — старый путь одним вызовом.
        audio = _run_tts({"text": _ensure_stop(text), "language": language,
                          "speaker": speakers_list[0]})
        audio = _postprocess(audio, SAMPLE_RATE)

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format='WAV')
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
    """Проставляет ударения (RUAccent) и возвращает текст. Приложение вызывает
    это для ВСЕХ движков (Edge/Google/системного/своего), чтобы произношение
    было одинаково корректным. Если RUAccent выключен — вернёт текст как есть."""
    require_auth(authorization)
    return {"text": accentize(request.text or "")}


@app.post("/v1/audio/speech")
async def openai_speech(request: SpeechRequest, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    try:
        speaker = resolve_speaker(request.voice)
        audio_buf = synthesize(request.input, speaker_wav_path=speaker, language=request.language)
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
    # accent=true — проставить ударения на сервере. Приложение обычно уже
    # прислало текст с ударениями (через /accent), поэтому по умолчанию False,
    # чтобы не проставлять дважды.
    if accent:
        text = accentize(text)
    speaker_path = None
    tmp_upload = None
    try:
        if speaker_wav:
            suffix = os.path.splitext(speaker_wav.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await speaker_wav.read())
                tmp_upload = tmp.name
            speaker_path = tmp_upload
        else:
            speaker_path = resolve_speaker(voice)

        audio_buf = synthesize(text, speaker_wav_path=speaker_path, language=language)
        return StreamingResponse(audio_buf, media_type="audio/wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if tmp_upload and os.path.exists(tmp_upload):
            os.unlink(tmp_upload)


@app.get("/voices")
async def list_voices(authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    return {"voices": list_voice_names(), "default": DEFAULT_SPEAKER}


@app.post("/voices/upload")
async def upload_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """Сохраняет эталонный образец голоса. Рекомендуется 6–20 секунд чистой
    речи без шума/музыки. Хранится как voices/<name>.wav."""
    require_auth(authorization)
    import subprocess
    dest = voice_path(name)
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1] or ".bin", delete=False) as tmp:
        tmp.write(data)
        tmp_in = tmp.name
    try:
        # Перекодируем в WAV 24 кГц моно через ffmpeg — надёжно для любого
        # присланного формата (m4a/aac/mp3/ogg/wav), которые libsndfile не читает.
        cmd = ["ffmpeg", "-y", "-i", tmp_in, "-ar", "24000", "-ac", "1", dest]
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
        "tts_lib_version": TTS_LIB_VERSION,
        "model": "xtts_v2",
        "custom_model_dir": XTTS_MODEL_DIR or None,
        "device": device,
        "auth": bool(API_KEY),
        "ruaccent": bool(_accentizer),
        "accent_response_mode": STRESS_MODE,
        "synth_stress": SYNTH_STRESS,
        "voices": list_voice_names(),
        "default_voice": DEFAULT_SPEAKER,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
    }
