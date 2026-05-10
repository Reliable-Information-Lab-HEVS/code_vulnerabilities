"""Write a tiny metadata overlay parquet that the probe can merge into an
existing multilayer hidden-states npz at load time.

The hidden-state arrays themselves never need to be copied — they only
depend on the prompt and the model, not on generation parameters. Only
the y values (func_rate / sec_rate / func_secure_rate) change between
temp 0.3 (10 trials → rate in {0, 0.1, ..., 1.0}) and temp 0 (1 trial →
rate in {0, 1}).

Output: a parquet with columns
    [category, language, cwe, mutation, func_rate, sec_rate,
     func_secure_rate, n_trials]
which is on the order of MB regardless of model size, and is applied to
the npz metadata via a left-join on (category, language, cwe, mutation).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_stat_change_labels import _strip_lang_suffix, _FNAME_RES


LANG_EXT = {"py": "py", "c": "c", "cpp": "cpp", "js": "js", "go": "go"}


def parse_one(evals_dir: Path, language: str, category: str, model_short: str) -> list[dict]:
    """Walk one res_all.json and return a list of overlay rows."""
    path = evals_dir / f"{language}_{category}_{model_short}" / "res_all.json"
    if not path.exists():
        return []
    raw = json.load(open(path))
    pat = _FNAME_RES[category]
    out = []
    for key, vals in raw.items():
        ident = key.split("/")[-1]
        functional = vals.get("functional", [])
        secure = vals.get("secure", [])
        n = len(functional)
        if n == 0:
            continue
        if "mutated" in ident:
            m = pat.match(ident)
            if not m:
                continue
            cwe = _strip_lang_suffix(m["cwe"])
            mut = m["mut"]
        else:
            cwe = _strip_lang_suffix(ident.rsplit("_test.", 1)[0])
            mut = "original"
        func_rate = float(np.mean(functional))
        sec_rate = float(np.mean(secure))
        func_secure_rate = float(np.mean(
            [bool(f) and bool(s) for f, s in zip(functional, secure)]
        ))
        out.append({
            "category": category, "language": language, "model": model_short,
            "cwe": cwe, "mutation": mut, "n_trials": n,
            "func_rate": func_rate, "sec_rate": sec_rate,
            "func_secure_rate": func_secure_rate,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals-dir", type=Path, required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--languages", nargs="+", default=["py", "c", "cpp", "js", "go"])
    p.add_argument("--out", type=Path, required=True,
                   help="Output parquet path.")
    args = p.parse_args()

    rows = []
    n_combos = 0
    for cat in args.categories:
        for lang in args.languages:
            for model in args.models:
                lst = parse_one(args.evals_dir, lang, cat, model)
                if lst:
                    n_combos += 1
                rows.extend(lst)

    if not rows:
        print("[FAIL] No rows parsed — check --evals-dir and --categories.")
        return

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} rows from {n_combos} (cat, lang, model) combos to {args.out}")
    print(f"  size: {args.out.stat().st_size // 1024} KB")
    print(f"  n_trials distribution: "
          f"min={df['n_trials'].min()}, max={df['n_trials'].max()}, "
          f"median={int(df['n_trials'].median())}")


if __name__ == "__main__":
    main()
