"""Test isolation that has to happen before any bench module is imported.

unittest imports this package before any test module, and bench.settings
reads its paths and profile from the environment once, at import. So every
test process gets a fresh temp root for the env file, run data and ledger;
provider keys that cannot bill anyone (a real key in the launching shell is
replaced, never used); and no .env loading by Harbor, LiteLLM or dotenv.

Run: .venv/Scripts/python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

if "bench.settings" in sys.modules:
    raise RuntimeError("bench was imported before tests/__init__.py; its paths point at real data")

TEST_ROOT = Path(tempfile.mkdtemp(prefix="bench-tests-"))
ENV_FILE = TEST_ROOT / "test.env"
ENV_FILE.write_text("", encoding="utf-8")

FAKE_KEYS = {
    "ANTHROPIC_API_KEY": "sk-test-1111111111111111abcd",
    "OPENROUTER_API_KEY": "sk-test-0000000000000000abcd",
}

# Anything that looks like a credential or an endpoint override in the
# launching shell is dropped, so a test can neither use nor leak it.
_CREDENTIAL = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|_BASE_URL|_API_BASE)$", re.I)
for _name in [n for n in os.environ if _CREDENTIAL.search(n)]:
    del os.environ[_name]
os.environ.update(FAKE_KEYS)

os.environ["BENCH_ENV_FILE"] = str(ENV_FILE)
os.environ["BENCH_DATA_DIR"] = str(TEST_ROOT / "data")
os.environ["BENCH_LEDGER"] = str(TEST_ROOT / "ledger" / "bench.sqlite")
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.pop("BENCH_PROFILE", None)

atexit.register(shutil.rmtree, TEST_ROOT, ignore_errors=True)
