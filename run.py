#!/usr/bin/env python3
"""
Entry point for the PD-Deep Precision Suite.

    conda activate pd_strat
    python run.py                       # uses ./config.yaml
    python run.py --data_dir S:/AMP-PD  # override the data folder
    python run.py --skip_robustness     # faster first pass
    python run.py --report_only         # regenerate SUMMARY_REPORT.md
    python run.py --help

Console output is also written to results/run_log.txt.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _Tee(io.TextIOBase):
    """Duplicate stdout/stderr into a log file."""

    def __init__(self, stream, fh):
        self._s, self._fh = stream, fh

    def write(self, s):
        self._s.write(s)
        self._fh.write(s)
        return len(s)

    def flush(self):
        self._s.flush()
        self._fh.flush()

    @property
    def encoding(self):
        return getattr(self._s, "encoding", "utf-8")


def _utf8_console():
    # Windows consoles default to a legacy code page when output is
    # redirected; force UTF-8 so unicode in log lines never crashes the run.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _utf8_console()
    from pd_strat.config import OUT
    from pd_strat.main import main as _main

    log_path = OUT / "run_log.txt"
    with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
        so, se = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(so, fh), _Tee(se, fh)
        try:
            fh.write("\n" + "#" * 70 + "\n")
            fh.write(f"# {' '.join(sys.argv)}\n")
            return _main()
        finally:
            sys.stdout, sys.stderr = so, se
            print(f"[Log] {log_path}")


if __name__ == "__main__":
    main()
