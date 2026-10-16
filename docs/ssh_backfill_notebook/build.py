import json, re, subprocess, sys
from pathlib import Path
here = Path(__file__).parent
M = json.load(open(here / "measurements.json"))
# computed, not typed: how many tests the suite has
repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/SDSMT.LOCAL/bscott/projects/opym_local")
out = subprocess.run([sys.argv[2] if len(sys.argv) > 2 else "python3", "-m", "pytest", str(repo / "tests/test_stream_backfill.py"), "--collect-only", "-q", "-p", "no:cacheprovider"],
                     capture_output=True, text=True, env={"PYTHONPATH": str(repo / "src"), "PATH": "/usr/bin:/bin"})
M["tests"] = int(re.search(r"(\d+) tests? collected", out.stdout).group(1))
css = open(sys.argv[3]).read()
css = css[css.index("<style>"): css.index("</style>") + 8]
tpl = (here / "backfill.tmpl.html").read_text().replace("/*CSS*/", css).replace("/*DATA*/null", json.dumps(M).replace("</", "<\\/"))
(here / "notebook.html").write_text(tpl)
print("built; tests =", M["tests"])
