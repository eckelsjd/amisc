from pathlib import Path
import os
import shutil
import logging


def pytest_sessionfinish(session, exitstatus):
    """Delete all stray test build products."""
    files = os.listdir('.')
    logging.shutdown()

    for f in files:
        if Path(f).is_dir() and f.startswith('amisc_'):
            shutil.rmtree(f)
