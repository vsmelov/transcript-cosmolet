"""Скачивание записи Plaud по presigned_url в inbox (URL передаётся аргументом)."""
import sys
import urllib.request
from pathlib import Path

url, name = sys.argv[1], sys.argv[2]
dest = Path(r"C:\Users\v\PycharmProjects\claude-workspace\cosmolet\data\inbox") / name
dest.parent.mkdir(parents=True, exist_ok=True)
# presigned S3 — мимо системного прокси (он ломает такие запросы)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(url, timeout=300) as r, open(dest, "wb") as f:
    f.write(r.read())
print(f"OK {dest} {dest.stat().st_size/1024/1024:.1f} MB")
