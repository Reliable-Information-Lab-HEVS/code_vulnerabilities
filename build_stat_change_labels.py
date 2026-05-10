"""Build stat_change_labels.json from a folder of res_all.json eval files.

Replicates the Fisher-exact + Benjamini-Hochberg pipeline that previously
lived in analysis.ipynb (cells 1, 5, 6). For each (cat, lang, model, cwe,
mutation) it tests whether the mutation's `functional` and `secure` rates
differ from the per-(lang, model, cwe) baseline (originals pooled across
all available categories — typically 30 obs at temp 0.3, 3 obs at temp 0).

Output:
    stat_change_labels.json — same shape as before:
        {"func": {"cat|lang|model|cwe|mut": 0/1, ...},
         "sec":  {"cat|lang|model|cwe|mut": 0/1, ...}}
    joint_tests.parquet      — full per-mutation table (optional but cheap to
                               keep; it's what populates the JSON labels and
                               is useful for any downstream analysis).

For binary (n=1) outcomes Fisher is run as usual; statistical power is
naturally lower with n=1 baseline=3 — fewer mutations will be flagged
significant. That's correct behaviour, not a bug. If you want a looser
"any change" filter for the probe, use the new `frac_*_diff` columns in
joint_tests.parquet (raw-difference fraction, no statistical test).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# Regex per category — same as compute_embeddings._FNAME_RES, with tolerant
# `(?:_<lang>)?` so we work on languages whose filenames omit the lang infix.
# Optional `_v\d+` matches the evals_bigdata scheme where each mutation has
# six prompt-version variants; v is folded into `mut` so each variant is its
# own row (otherwise the six variants would collide on the same mut key).
# Old (no v-tag) names parse with mut='0_1'; new names with mut='0_1_v1'.
_FNAME_RES = {
    "mutated_token_replacement":  re.compile(
        r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_token_(?P<mut>\d+(?:_v\d+)?)_test\.[a-z]+$"),
    "mutated_prompts_character":  re.compile(
        r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_(?P<mut>\d+_\d+(?:_v\d+)?)_test\.[a-z]+$"),
    "mutated3_prompts_character": re.compile(
        r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_(?P<mut>\d+_\d+(?:_v\d+)?)_test\.[a-z]+$"),
}


def _strip_lang_suffix(cwe_code: str) -> str:
    for s in ("_c", "_cpp", "_go", "_js", "_py"):
        if cwe_code.endswith(s):
            return cwe_code[:-len(s)]
    return cwe_code


def parse_eval(evals_dir: Path, language: str, category: str, model_short: str) -> dict:
    """Return {cwe: {mut_key: {functional: [bool], secure: [bool]}}}.

    `mut_key='original'` for the unmutated original; mutation keys are the
    `mut` group from the regex (e.g. `'5'` for token-replacement, `'0_1'` for
    character mutations).
    """
    path = evals_dir / f"{language}_{category}_{model_short}" / "res_all.json"
    if not path.exists():
        return {}
    raw = json.load(open(path))
    pat = _FNAME_RES[category]
    out: dict = defaultdict(dict)
    for key, vals in raw.items():
        ident = key.split("/")[-1]
        functional = vals.get("functional", [])
        secure = vals.get("secure", [])
        if not functional:
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
        out[cwe][mut] = {
            "functional": [bool(x) for x in functional],
            "secure":     [bool(x) for x in secure],
        }
    return dict(out)


def load_all(evals_dir: Path, models: list[str], languages: list[str],
             categories: list[str]) -> dict:
    """Returns nested dict: {category: {language: {model: {cwe: {mut: data}}}}}."""
    print(f"Loading evals from {evals_dir} ...", flush=True)
    out = {}
    n_loaded = n_missing = 0
    for cat in categories:
        out[cat] = {}
        for lang in languages:
            out[cat][lang] = {}
            for model in models:
                d = parse_eval(evals_dir, lang, cat, model)
                if d:
                    out[cat][lang][model] = d
                    n_loaded += 1
                else:
                    n_missing += 1
    print(f"  loaded {n_loaded} (cat, lang, model) combos; "
          f"missing {n_missing}", flush=True)
    return out


def _detect_n_trials(by_cat: dict) -> int:
    """Return the max number of trials per mutation observed anywhere in the
    data. n=1 means deterministic (temp=0) — no statistical test needed."""
    n_max = 0
    for langs in by_cat.values():
        for models in langs.values():
            for cwes in models.values():
                for muts in cwes.values():
                    for data in muts.values():
                        n_max = max(n_max, len(data.get("functional", [])))
    return n_max


def compute_pairwise_tests(by_cat: dict, metric: str,
                           binary: bool = False) -> pd.DataFrame:
    """For each (cat, lang, model, cwe, mut), test whether the mutation's
    `metric` rate differs from the per-(lang, model, cwe) original baseline
    pooled across all available categories.

    With multiple trials per mutation we run Fisher exact (two-sided) and
    populate a real `pvalue`. With deterministic n=1 outcomes (`binary=True`)
    there is no sampling distribution: a mutation is "significant" iff its
    outcome differs from the same-category original (or, if that's missing,
    from the pooled-baseline majority). p-value is set to 0 for changed
    mutations and 1 for unchanged ones, so the downstream FDR step preserves
    these flags exactly.
    """
    higher_is_worse = metric == "func_insecure"

    def _vals(d):
        if metric == "func_insecure":
            return [f and not s for f, s in zip(d["functional"], d["secure"])]
        return d[metric]

    # Pool original observations across categories per (lang, model, cwe).
    pooled_orig: dict = {}
    same_cat_orig: dict = {}  # (cat, lang, model, cwe) -> list[bool]
    for cat, langs in by_cat.items():
        for lang, models in langs.items():
            for model, cwes in models.items():
                for cwe, muts in cwes.items():
                    if "original" not in muts:
                        continue
                    vals = _vals(muts["original"])
                    same_cat_orig[(cat, lang, model, cwe)] = vals
                    pooled_orig.setdefault((lang, model, cwe), []).extend(vals)

    rows = []
    for cat, langs in by_cat.items():
        for lang, models in langs.items():
            for model, cwes in models.items():
                for cwe, muts in cwes.items():
                    key = (lang, model, cwe)
                    if key not in pooled_orig:
                        continue
                    base = pooled_orig[key]
                    b_pos, b_total = sum(base), len(base)
                    b_rate = b_pos / b_total

                    for mut_key, mut_data in muts.items():
                        if mut_key == "original":
                            continue
                        mvals = _vals(mut_data)
                        m_pos, m_total = sum(mvals), len(mvals)
                        m_rate = m_pos / m_total

                        if binary:
                            # Same-category baseline if present, else pooled majority.
                            sc = same_cat_orig.get((cat, lang, model, cwe))
                            if sc:
                                ref = bool(round(sum(sc) / len(sc)))
                            else:
                                ref = bool(round(b_rate))
                            changed = bool(mvals[0]) != ref
                            pval = 0.0 if changed else 1.0
                            odds = float("nan")
                        else:
                            table = np.array([
                                [b_pos, b_total - b_pos],
                                [m_pos, m_total - m_pos],
                            ])
                            odds, pval = fisher_exact(table, alternative="two-sided")

                        diff = m_rate - b_rate
                        rows.append({
                            "category": cat, "language": lang, "model": model,
                            "cwe": cwe, "mutation": mut_key,
                            "metric": metric,
                            "baseline_positive": b_pos, "baseline_total": b_total,
                            "mutation_positive": m_pos, "mutation_total": m_total,
                            "baseline_rate": b_rate, "mutation_rate": m_rate,
                            "rate_diff": diff,
                            "security_diff": -diff if higher_is_worse else diff,
                            "odds_ratio": odds, "pvalue": pval,
                        })
    if not rows:
        # Empty input (e.g. no parseable mutations or no matching baselines)
        # — return a DataFrame with the expected schema so downstream code
        # can index df["pvalue"] etc. without KeyError.
        return pd.DataFrame(columns=[
            "category", "language", "model", "cwe", "mutation", "metric",
            "baseline_positive", "baseline_total",
            "mutation_positive", "mutation_total",
            "baseline_rate", "mutation_rate",
            "rate_diff", "security_diff", "odds_ratio", "pvalue",
        ])
    return pd.DataFrame(rows)


def apply_fdr(df: pd.DataFrame, group_cols: list[str], alpha: float = 0.05,
              method: str = "fdr_bh") -> pd.DataFrame:
    df = df.copy()
    df["pvalue_corrected"] = np.nan
    df["significant"] = False
    for _, gdf in df.groupby(group_cols):
        if not len(gdf):
            continue
        reject, pcorr, _, _ = multipletests(gdf["pvalue"].values, alpha=alpha, method=method)
        df.loc[gdf.index, "pvalue_corrected"] = pcorr
        df.loc[gdf.index, "significant"] = reject
    return df


def compute_joint_tests(by_cat: dict, alpha: float = 0.05) -> pd.DataFrame:
    metrics = ["functional", "secure", "func_insecure"]
    n_trials = _detect_n_trials(by_cat)
    binary = n_trials <= 1
    if binary:
        print(f"  detected deterministic outcomes (max trials per mutation = "
              f"{n_trials}); using direct difference instead of Fisher exact.",
              flush=True)
    else:
        print(f"  detected multi-trial outcomes (max = {n_trials}); using "
              f"Fisher exact + FDR.", flush=True)

    per_metric = {}
    for m in metrics:
        df = compute_pairwise_tests(by_cat, metric=m, binary=binary)
        if binary:
            # No multiple-comparison correction needed for a deterministic flip.
            df = df.copy()
            df["pvalue_corrected"] = df["pvalue"]
            df["significant"] = df["pvalue"] < alpha
        else:
            df = apply_fdr(df, group_cols=["model", "cwe", "language"], alpha=alpha)
        per_metric[m] = df

    keys = ["category", "language", "model", "cwe", "mutation"]
    merged = per_metric["func_insecure"][keys + [
        "baseline_rate", "mutation_rate", "rate_diff", "security_diff",
        "pvalue", "pvalue_corrected", "significant",
    ]].rename(columns={
        "baseline_rate": "fi_baseline_rate", "mutation_rate": "fi_mutation_rate",
        "rate_diff": "fi_rate_diff", "security_diff": "fi_security_diff",
        "pvalue": "fi_pvalue", "pvalue_corrected": "fi_pvalue_corrected",
        "significant": "fi_significant",
    })
    for metric in ("functional", "secure"):
        suffix = "func" if metric == "functional" else "sec"
        sub = per_metric[metric][keys + [
            "baseline_rate", "mutation_rate", "rate_diff",
            "pvalue", "pvalue_corrected", "significant",
        ]].rename(columns={
            "baseline_rate": f"{suffix}_baseline_rate",
            "mutation_rate": f"{suffix}_mutation_rate",
            "rate_diff": f"{suffix}_rate_diff",
            "pvalue": f"{suffix}_pvalue",
            "pvalue_corrected": f"{suffix}_pvalue_corrected",
            "significant": f"{suffix}_significant",
        })
        merged = merged.merge(sub, on=keys, how="left")
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evals-dir", type=Path, required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=[
        "Qwen3-Coder-30B-A3B-Instruct",
        "deepseek-coder-33b-instruct",
        "CodeLlama-70b-Instruct-hf",
    ])
    p.add_argument("--languages", nargs="+", default=["py", "c", "cpp", "js", "go"])
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-parquet", type=Path, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    by_cat = load_all(args.evals_dir, args.models, args.languages, args.categories)
    if not any(by_cat[c] for c in by_cat):
        print("No evals loaded — aborting.")
        return

    print("Running Fisher exact + FDR-BH per (model, cwe, language) ...", flush=True)
    jdf = compute_joint_tests(by_cat, alpha=args.alpha)
    print(f"  total rows: {len(jdf)}, "
          f"func_significant: {int(jdf['func_significant'].sum())}, "
          f"sec_significant: {int(jdf['sec_significant'].sum())}, "
          f"fi_significant: {int(jdf['fi_significant'].sum())}", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    func_labels = {
        "|".join([r["category"], r["language"], r["model"], r["cwe"], r["mutation"]]):
            int(bool(r["func_significant"])) for _, r in jdf.iterrows()
    }
    sec_labels = {
        "|".join([r["category"], r["language"], r["model"], r["cwe"], r["mutation"]]):
            int(bool(r["sec_significant"])) for _, r in jdf.iterrows()
    }
    json.dump({"func": func_labels, "sec": sec_labels}, open(args.out_json, "w"))
    print(f"Saved labels JSON: {args.out_json}", flush=True)

    if args.out_parquet is not None:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        jdf.to_parquet(args.out_parquet, index=False)
        print(f"Saved joint-tests parquet: {args.out_parquet}", flush=True)


if __name__ == "__main__":
    main()
