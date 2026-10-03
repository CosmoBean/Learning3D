#!/bin/bash
# usage: md2html.sh report.md report.html
# Renders a report to a standalone page: local images are inlined as base64 so the html works on its own.
set -e
python3 - "$1" "$2" <<'EOF'
import base64, mimetypes, os, re, sys
import markdown

src, dst = sys.argv[1:]
body = markdown.markdown(open(src).read(), extensions=["tables", "fenced_code"])

def inline(m):
    p = os.path.join(os.path.dirname(src), m.group(1))
    if not os.path.isfile(p):
        return m.group(0)
    data = base64.b64encode(open(p, "rb").read()).decode()
    return f'src="data:{mimetypes.guess_type(p)[0]};base64,{data}"'

body = re.sub(r'src="([^":]+)"', inline, body)
title = re.search(r"<h1>(.*?)</h1>", body)
open(dst, "w").write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title.group(1) if title else os.path.basename(src)}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ max-width: 860px; margin: 2rem auto; padding: 0 1.2rem;
         font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         line-height: 1.6; color: #111; background: #fff; }}
  h1, h2, h3 {{ line-height: 1.25; color: #000; }}
  h1 {{ border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }}
  a {{ color: #111; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #eee; margin: .3rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: .5rem .7rem; text-align: left; vertical-align: top; }}
  th {{ background: #f2f2f2; }}
  code {{ background: #f2f2f2; padding: .1rem .3rem; border-radius: 3px; font-size: .9em; }}
  pre {{ background: #f6f6f6; border: 1px solid #eee; padding: 1rem; border-radius: 4px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
""")
print("wrote", dst)
EOF
