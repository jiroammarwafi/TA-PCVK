"""Konversi dokumen Markdown menjadi PDF memakai mistune + headless Chrome.

Tidak butuh pandoc/LaTeX. Cukup Chrome atau Edge yang sudah terpasang di Windows.

Contoh:
    python scripts/build_pdf.py docs/naskah-video-progres.md
    python scripts/build_pdf.py docs/bab3-dataset-dan-praproses.md -o docs/bab3.pdf
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

CSS = """
@page { size: A4; margin: 15mm 15mm 15mm 15mm; }

* { box-sizing: border-box; }

body {
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  font-size: 10pt;
  line-height: 1.48;
  color: #1a1a1a;
  margin: 0;
}

h1 {
  font-size: 20pt;
  line-height: 1.25;
  margin: 0 0 6pt;
  color: #10243f;
  border-bottom: 3pt solid #1f4e79;
  padding-bottom: 8pt;
}

h2 {
  font-size: 14pt;
  margin: 16pt 0 7pt;
  color: #1f4e79;
  border-bottom: 1pt solid #c9d6e4;
  padding-bottom: 4pt;
  break-after: avoid;
}

h3 {
  font-size: 12pt;
  margin: 12pt 0 5pt;
  color: #24486b;
  break-after: avoid;
}

h1 + p, h2 + p, h3 + p { margin-top: 0; }

p { margin: 0 0 6pt; text-align: justify; orphans: 3; widows: 3; }

strong { color: #10243f; }

em { color: #333; }

a { color: #1f4e79; text-decoration: none; }

hr {
  border: none;
  border-top: 1pt solid #dde5ee;
  margin: 14pt 0;
}

blockquote {
  margin: 8pt 0 10pt;
  padding: 8pt 12pt;
  background: #f3f7fb;
  border-left: 3pt solid #1f4e79;
  color: #24313d;
  break-inside: avoid;
}
blockquote p { margin: 0 0 4pt; }
blockquote p:last-child { margin-bottom: 0; }

ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin: 0 0 3pt; }

table {
  width: 100%;
  border-collapse: collapse;
  margin: 8pt 0 12pt;
  font-size: 9.5pt;
  break-inside: avoid;}

th {
  background: #1f4e79;
  color: #ffffff;
  font-weight: 600;
  text-align: left;
  padding: 5pt 7pt;
  border: 1pt solid #1f4e79;
}

td {
  padding: 4.5pt 7pt;
  border: 1pt solid #cdd8e3;
  vertical-align: top;
}

tbody tr:nth-child(even) td { background: #f6f9fc; }

code {
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 9pt;
  background: #eef2f7;
  padding: 0.5pt 3pt;
  border-radius: 2pt;
  color: #123a5e;
}

pre {
  background: #f5f7fa;
  border: 1pt solid #dde5ee;
  border-left: 3pt solid #6b8cb0;
  border-radius: 3pt;
  padding: 8pt 10pt;
  margin: 8pt 0 10pt;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 9pt; color: #16324f; }

/* Penanda halaman: <div class="segment"> memulai halaman baru sebelum elemen berikutnya. */
.segment { break-before: page; height: 0; }
.pagebreak { break-after: page; height: 0; }

.footer-note {
  margin-top: 18pt;
  padding-top: 6pt;
  border-top: 1pt solid #dde5ee;
  font-size: 8.5pt;
  color: #6b7785;
  text-align: center;
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
<div class="footer-note">{footer}</div>
</body>
</html>
"""

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "chrome",
    "msedge",
]


def find_browser() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit(
        "Chrome/Edge tidak ditemukan. Pasang Chrome, atau tambahkan ke PATH."
    )


def markdown_to_html(text: str) -> str:
    renderer = mistune.create_markdown(
        escape=False,
        plugins=["table", "strikethrough", "url"],
    )
    return renderer(text)


def build(md_path: Path, pdf_path: Path, title: str | None) -> Path:
    if not md_path.exists():
        raise SystemExit(f"File tidak ditemukan: {md_path}")

    pdf_path = pdf_path.resolve()
    source = md_path.read_text(encoding="utf-8")
    body = markdown_to_html(source)

    page_title = title or md_path.stem.replace("-", " ").title()
    footer = html.escape(f"{md_path.name}  ·  dibuat otomatis dari Markdown")
    page = HTML_TEMPLATE.format(
        title=html.escape(page_title),
        css=CSS,
        body=body,
        footer=footer,
    )

    with tempfile.TemporaryDirectory(prefix="md2pdf-") as tmp:
        tmp_html = Path(tmp) / "page.html"
        tmp_html.write_text(page, encoding="utf-8")

        browser = find_browser()
        profile = Path(tmp) / "profile"
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=5000",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path}",
            tmp_html.as_uri(),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )

    if not pdf_path.exists():
        detail = (result.stderr or result.stdout or "").strip()
        raise SystemExit(f"Gagal membuat PDF.\n{detail}")

    return pdf_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="File Markdown sumber")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Path PDF keluaran"
    )
    parser.add_argument("--title", default=None, help="Judul dokumen di metadata")
    args = parser.parse_args(argv)

    pdf_path = args.output or args.source.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    out = build(args.source, pdf_path, args.title)
    size_kb = out.stat().st_size / 1024
    print(f"OK  {out}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
