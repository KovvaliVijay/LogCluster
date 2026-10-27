#!/usr/bin/env bash

set -euo pipefail

mkdir -p docs/_build/html

if command -v pandoc >/dev/null 2>&1; then
  pandoc README.md -o docs/_build/html/index.html
else
  python3 - <<'PY'
from html import escape
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")
html = "<html><body><pre>" + escape(readme) + "</pre></body></html>\n"
Path("docs/_build/html/index.html").write_text(html, encoding="utf-8")
PY
fi
