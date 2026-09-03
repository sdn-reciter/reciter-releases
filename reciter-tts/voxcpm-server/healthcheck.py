# Docker HEALTHCHECK: жив ли /health внутри контейнера (порт 8002 внутренний).
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8002/health", timeout=8) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
