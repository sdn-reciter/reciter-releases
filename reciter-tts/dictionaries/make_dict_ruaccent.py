#!/usr/bin/env python3
"""
Export the RUAccent *dictionary* (no neural model) into the compact asset that
the reader loads on-device: app/src/dev/assets/ruaccent_dict_ruaccent.dict

CC BY-NC-ND 4.0 data — dev flavor ONLY, never ship in a public build.

Run it in Google Colab (or any machine with `pip install ruaccent`), then drop
the produced file into app/src/dev/assets/. See docs/reciter-tts/accent-dictionary.md.

Output format (matches AccentDictionary.kt):
  • UTF-8, lines separated by '\n', sorted ASCENDING by raw UTF-8 BYTES;
  • each line is  key<TAB>pos  where
        key = lowercased word WITHOUT accents (as printed in a book),
        pos = 0-based code-point index of the stressed vowel
              (the app inserts U+0301 itself).
Sorting by bytes == sorting by code point, so the app can binary-search the
memory-mapped (gzip-inflated) file directly.

Why only the dictionary: the neural omograph model needs a full ML runtime and
is too heavy for a phone. The dictionary is a plain lookup — RAM ~0, lookups in
microseconds. Homographs fall back to the dictionary's default stress.
"""

import subprocess
import sys

ACUTE = "́"  # комбинируемый акут (ударение)
# Словарь для личного использования (CC BY-NC-ND) — в app/src/dev/assets/.
OUT = "ruaccent_dict_ruaccent.dict"


def ensure_ruaccent():
    """Imports RUAccent, pip-installing it on first run (Colab-friendly)."""
    try:
        import ruaccent  # noqa: F401
    except ModuleNotFoundError:
        print("ruaccent not found — installing...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ruaccent"])
        import ruaccent  # noqa: F401


def strip_accents(s: str) -> str:
    """Removes combining acute marks and lowercases — yields the lookup key."""
    return "".join(c for c in s if c != ACUTE).lower()


VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def plus_to_acute(s: str):
    """Converts a RUAccent "+"-marked word to a word with U+0301 after the
    stressed vowel. RUAccent puts "+" right AFTER the stressed vowel, so a plain
    replace works; we still handle a "+"-before-vowel layout defensively.
    Returns None if there's no usable stress mark."""
    if "+" not in s:
        return None
    i = s.index("+")
    before = s[i - 1] if i > 0 else ""
    after = s[i + 1] if i + 1 < len(s) else ""
    stripped = s.replace("+", "", 1)
    pos = None
    if before in VOWELS:        # "приве+т" — vowel just before the plus
        pos = i - 1
    elif after in VOWELS:       # "+ве" style — vowel just after the plus
        pos = i
    if pos is None:
        return None
    return stripped[: pos + 1] + ACUTE + stripped[pos + 1 :]


def _iter_dict_files():
    """Yields RUAccent dictionary JSON(.gz) files across likely locations."""
    import os
    roots = [
        os.path.dirname(sys.modules["ruaccent"].__file__),
        os.getcwd(),
        os.path.expanduser("~/.cache/huggingface"),
        "/root/.cache/huggingface",
    ]
    seen_paths = set()
    for base in roots:
        if not base or not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for f in files:
                low = f.lower()
                if low.startswith("accents") and (low.endswith(".json") or low.endswith(".json.gz")):
                    p = os.path.join(root, f)
                    if p not in seen_paths:
                        seen_paths.add(p)
                        yield p


def _load_json_any(path):
    import gzip
    import json
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def stress_pos(accented: str):
    """Given a "+"- or U+0301-marked word, returns (key, pos) where key is the
    lowercased un-accented word and pos is the 0-based CODE-POINT index of the
    stressed vowel. Returns None if unusable."""
    if "+" in accented:
        val = plus_to_acute(accented)
    elif ACUTE in accented:
        val = accented
    else:
        return None
    if not val or ACUTE not in val:
        return None
    i = val.index(ACUTE)          # комбинируемый знак стоит СРАЗУ ПОСЛЕ гласной
    pos = i - 1
    key = val.replace(ACUTE, "").lower()
    if pos < 0 or pos >= len(key) or key[pos] not in VOWELS:
        return None
    if not all(c.isalpha() for c in key):
        return None
    return key, pos


def _harvest(data, seen, sampled):
    """Harvests {key: stress_pos} from one loaded dictionary (a dict)."""
    if not isinstance(data, dict):
        return 0
    added = 0
    for word, accented in data.items():
        if isinstance(accented, list):
            accented = next((x for x in accented if isinstance(x, str)), None)
        if not isinstance(accented, str):
            continue
        parsed = stress_pos(accented)
        if parsed is None:
            continue
        key, pos = parsed
        if len(sampled) < 8:
            sampled.append((word, accented, f"{key}#{pos}"))
        if key not in seen:
            seen[key] = pos
            added += 1
    return added


def collect_pairs():
    """Returns {key: accented} harvested from RUAccent's on-disk dictionaries.

    Primary source is `dictionary/accents.json.gz` (the big lookup); we also
    fold in any other `accents*.json(.gz)`. The neural models are ignored.
    """
    ensure_ruaccent()
    from ruaccent import RUAccent  # noqa

    accentizer = RUAccent()
    # Скачивает словари + модели в пакет / кеш HF.
    accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True)

    seen = {}
    sampled = []
    files = sorted(_iter_dict_files(), key=lambda p: (0 if "accents.json" in p.lower() else 1, p))
    for path in files:
        try:
            data = _load_json_any(path)
        except Exception as e:
            print(f"  skip {path}: {e}", file=sys.stderr)
            continue
        n = _harvest(data, seen, sampled)
        print(f"  {path}: +{n}", file=sys.stderr)

    if sampled:
        print("Sample (word | raw | accented):", file=sys.stderr)
        for w, raw, v in sampled[:8]:
            print(f"    {w} | {raw} | {v}", file=sys.stderr)
    return seen


def main():
    import gzip
    import os

    pairs = collect_pairs()
    print(f"Collected {len(pairs)} word forms", file=sys.stderr)
    if not pairs:
        print("No pairs collected — inspect RUAccent's data layout.", file=sys.stderr)
        sys.exit(1)

    # Сортировка по сырым UTF-8-байтам — чтобы побайтовый бинпоиск в приложении был валиден.
    # Каждая строка — `key\t<pos>`, где pos — десятичный индекс код-поинта
    # ударной гласной — гораздо компактнее, чем хранить слово с ударением; приложение
    # восстанавливает U+0301 в этой позиции. Всё gzip-сжато;
    # приложение распаковывает один раз в приватное хранилище и mmap-ит его.
    items = sorted(pairs.items(), key=lambda kv: kv[0].encode("utf-8"))
    raw_bytes = 0
    with gzip.open(OUT, "wb", compresslevel=9) as out:
        for key, pos in items:
            if "\t" in key or "\n" in key:
                continue
            line = f"{key}\t{pos}\n".encode("utf-8")
            raw_bytes += len(line)
            out.write(line)

    print(
        f"Wrote {OUT}: {os.path.getsize(OUT)/1_000_000:.1f} MB gzip "
        f"(uncompressed {raw_bytes/1_000_000:.1f} MB, {len(items)} forms)",
        file=sys.stderr,
    )
    _try_colab_download(OUT)


def _try_colab_download(path):
    """Auto-starts a browser download when running in Google Colab; no-op
    outside Colab. On failure prints how to grab the file manually instead of
    silently doing nothing."""
    try:
        from google.colab import files  # type: ignore
    except ImportError:
        return  # не Colab — файл остаётся рядом со скриптом
    try:
        print(f"Colab: скачиваю {path}…", file=sys.stderr)
        files.download(path)
    except Exception as e:
        print(f"Colab: files.download не сработал ({e}).\n"
              f"Скачай вручную: панель «Файлы» слева → {path} → ⋮ → Download.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
