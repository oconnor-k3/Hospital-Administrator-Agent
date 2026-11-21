"""Minimal runner so tests work without pytest: python tests/run_tests.py"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_env

failed = 0
for name in sorted(n for n in dir(test_env) if n.startswith("test_")):
    try:
        getattr(test_env, name)()
        print(f"PASS {name}")
    except Exception:
        failed += 1
        print(f"FAIL {name}")
        traceback.print_exc()
sys.exit(1 if failed else 0)
