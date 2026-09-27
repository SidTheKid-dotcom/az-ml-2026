"""Agent Y: assemble README.md = static header (README_head.md) + y_summary output + SHA256 of delivered files.
Usage: y_readme.py N HEAD.md DELIVER_DIR OUT.md"""
import subprocess, sys
from pathlib import Path
N, head, deliver, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
summary = subprocess.run([sys.executable, str(Path(__file__).with_name("y_summary.py")), N], capture_output=True,
                         text=True, check=True).stdout
sums = (deliver / "SHA256SUMS").read_text() if (deliver / "SHA256SUMS").exists() else "(SHA256SUMS missing)\n"
out.write_text(head.read_text().rstrip() + "\n\n" + summary.rstrip() + "\n\n## Files (SHA256, paths relative to s3://az-ml-new-20260926-533267274451/run27/Y/)\n```\n"
               + sums + "```\n")
print(out)
