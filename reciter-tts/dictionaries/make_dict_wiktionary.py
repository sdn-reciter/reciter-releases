#!/usr/bin/env python3
"""
Build the on-device stress dictionary from the RUSSIAN WIKTIONARY (via Tatu
Ylonen's wiktextract data on kaikki.org) instead of RUAccent.

Why: RUAccent's data is CC BY-NC-ND 4.0 (NonCommercial + NoDerivatives), so a
transformed dictionary can't be shipped in a public app. Wiktionary content is
**CC BY-SA** — производные РАЗРЕШЕНЫ, нужно лишь (a) указать Wiktionary
and (b) release the derived dictionary under the same CC BY-SA. That makes this
output legally distributable. See docs/licenses.md.

Output: `ruaccent_dict_wiktionary.dict` — the format AccentDictionary.kt
reads: gzip of UTF-8 lines `key\t<pos>` sorted by raw bytes, where key =
lowercased word without accents and pos = 0-based code-point index of the
stressed vowel. Drop it into app/src/main/assets/ (ships in all builds).

Usage (Colab):
    !python make_dict_wiktionary.py
It downloads the kaikki Russian extract (~770 MB) and writes ruaccent_dict_wiktionary.dict.
If the download URL 404s, grab the current "all senses" JSONL link from
https://kaikki.org/dictionary/Russian/ and pass it:
    !python make_dict_wiktionary.py --url <link>
Or point at an already-downloaded file:
    !python make_dict_wiktionary.py --file kaikki.org-dictionary-Russian.jsonl
"""

import argparse
import gzip
import json
import os
import sys
import urllib.request

ACUTE = "́"  # комбинируемый акут (ударение)
VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
# Словарь для публичного релиза (CC BY-SA) — в app/src/main/assets/.
OUT = "ruaccent_dict_wiktionary.dict"
DEFAULT_URL = "https://kaikki.org/dictionary/Russian/kaikki.org-dictionary-Russian.jsonl"


def stress_pos(accented: str):
    """(key, pos) from a U+0301-marked word, or None. key = lowercased word
    without the accent; pos = code-point index of the stressed vowel."""
    if ACUTE not in accented:
        return None
    i = accented.index(ACUTE)          # комбинируемый знак стоит ПОСЛЕ своей гласной
    pos = i - 1
    key = accented.replace(ACUTE, "").lower()
    if pos < 0 or pos >= len(key) or key[pos] not in VOWELS:
        return None
    if not all(c.isalpha() for c in key):
        return None
    # Пропуск слов с несколькими/побочными ударениями: только с одним ударением.
    if accented.count(ACUTE) != 1:
        return None
    return key, pos


def candidates(entry):
    """Yields accented strings from one kaikki JSON entry. The canonical stressed
    headword usually lives in forms[] (tag 'canonical'); we also scan every form
    and the top-level word just in case."""
    w = entry.get("word")
    if isinstance(w, str):
        yield w
    for f in entry.get("forms", []) or []:
        s = f.get("form")
        if isinstance(s, str):
            yield s
    # У части записей ударная форма — в разворотах head_templates.
    for ht in entry.get("head_templates", []) or []:
        exp = ht.get("expansion")
        if isinstance(exp, str):
            yield exp


def iter_entries(path):
    with open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def download(url, dest):
    print(f"Downloading {url}\n  → {dest} (this is ~770 MB, be patient)…", file=sys.stderr)
    def _hook(n, bs, total):
        if total > 0 and n % 500 == 0:
            pct = min(100, n * bs * 100 // total)
            print(f"\r  {pct}%", end="", file=sys.stderr)
    urllib.request.urlretrieve(url, dest, _hook)
    print("\n  done.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help="kaikki Russian JSONL URL")
    ap.add_argument("--file", default=None, help="already-downloaded JSONL file")
    # parse_known_args (не parse_args), чтобы вставка кода в ячейку Colab/Jupyter
    # и вызов main() работали — ядро подсовывает лишний «-f kernel.json»
    # в sys.argv, который обычный parse_args() отверг бы.
    args, _ignored = ap.parse_known_args()

    path = args.file
    if not path:
        path = "kaikki-russian.jsonl"
        if not os.path.exists(path):
            try:
                download(args.url, path)
            except Exception as e:
                print(f"Download failed: {e}\nGrab the JSONL link from "
                      f"https://kaikki.org/dictionary/Russian/ and pass --url/--file.",
                      file=sys.stderr)
                sys.exit(1)

    seen = {}
    sampled = []
    n_entries = 0
    for entry in iter_entries(path):
        n_entries += 1
        for cand in candidates(entry):
            parsed = stress_pos(cand)
            if parsed is None:
                continue
            key, pos = parsed
            if key not in seen:
                seen[key] = pos
                if len(sampled) < 8:
                    sampled.append((cand, f"{key}#{pos}"))
        if n_entries % 200000 == 0:
            print(f"  …{n_entries} entries, {len(seen)} forms", file=sys.stderr)

    print(f"Parsed {n_entries} entries → {len(seen)} stressed forms", file=sys.stderr)
    if sampled:
        print("Sample (accented | key#pos):", file=sys.stderr)
        for a, k in sampled:
            print(f"    {a} | {k}", file=sys.stderr)
    if not seen:
        print("No forms collected — check the JSONL schema / URL.", file=sys.stderr)
        sys.exit(1)

    items = sorted(seen.items(), key=lambda kv: kv[0].encode("utf-8"))
    raw = 0
    with gzip.open(OUT, "wb", compresslevel=9) as out:
        for key, pos in items:
            if "\t" in key or "\n" in key:
                continue
            line = f"{key}\t{pos}\n".encode("utf-8")
            raw += len(line)
            out.write(line)
    print(f"Wrote {OUT}: {os.path.getsize(OUT)/1_000_000:.1f} MB gzip "
          f"(uncompressed {raw/1_000_000:.1f} MB, {len(items)} forms)", file=sys.stderr)
    print("License: derived from Wiktionary — CC BY-SA. Ship the NOTICE "
          "(see docs/licenses.md) alongside the app.", file=sys.stderr)
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
