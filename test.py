"""Contest entry point, run from the archive root as the organisers' sample does (wiki/plan.md section 8):
python test.py --videos <dir> --results <dir>   (runs submission/test.py unchanged)"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "submission" / "test.py"), run_name="__main__")
