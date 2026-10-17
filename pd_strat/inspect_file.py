"""
Print the schema of any data file (CSV / TSV / parquet / xlsx) and, if it is an
Olink long table, how the pipeline would interpret it.

    python -m pd_strat.inspect_file "S:/AMP-PD/ppmi_proj293_plasma_screened_extended_npx_20251121.parquet"
    python -m pd_strat.inspect_file file1.csv file2.parquet --rows 5
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--rows", type=int, default=3)
    a = ap.parse_args(argv)
    from .olink_io import describe_table
    for p in a.paths:
        try:
            print(describe_table(p, a.rows))
        except Exception as e:
            print(f"{p}\n  ERROR {type(e).__name__}: {e}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
