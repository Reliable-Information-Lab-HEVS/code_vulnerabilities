"""Train probes on prompt embeddings to predict outcome rates per CWE.

For each (model, language, CWE) group, pool the mutations across all initial
mutation categories and use the prompt embeddings as features. Targets:
    - func_rate          (fraction of functional generations, in [0, 1])
    - func_secure_rate   (fraction of generations that are functional AND secure)

Baseline: predict the train-fold mean (the constant predictor). Reported as the
explicit MSE of that predictor on the test fold; R² of the baseline against the
test set is ~0 by construction.

Probes (all run with 5-fold CV):
    - linreg : Ridge regression
    - mlp1   : 1-hidden-layer MLP
    - mlp2   : 2-hidden-layer MLP

Outputs a CSV per model to embeddings/probe_results_<suffix>.csv.

Usage:
    python probe_embeddings.py
    python probe_embeddings.py --suffix initial --probes linreg mlp1 mlp2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Config ──────────────────────────────────────────────────────────────
EMB_DIR = Path("/cluster/raid/home/stea/CWEval/embeddings")
STAT_LABELS_PATH = EMB_DIR / "stat_change_labels.json"

MODELS = [
    "CodeLlama-70b-Instruct-hf",
    "deepseek-coder-33b-instruct",
    "Qwen3-Coder-30B-A3B-Instruct",
]
SHORT_NAMES = {
    "CodeLlama-70b-Instruct-hf": "CodeLlama-70b",
    "deepseek-coder-33b-instruct": "DeepSeek-33b",
    "Qwen3-Coder-30B-A3B-Instruct": "Qwen3-30b",
}

TARGETS = [
    ("func_rate", "Functional"),
    ("sec_rate",  "Secure"),
]
PROBES = ["linreg", "logreg", "mlp1", "mlp2"]
DEFAULT_PROBES = ["mlp2"]  # Phase 1 sweep: mlp2 with default dropout only.

CV_SPLITS    = 5
MIN_SAMPLES  = 25      # need at least this many mutations per group
RIDGE_ALPHA  = 1.0

MLP1_HIDDEN  = (256,)
MLP2_HIDDEN  = (512, 256)
MLP_DROPOUT  = 0.3
EPOCHS       = 200
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
LOGREG_C     = 1.0     # Inverse L2 strength for plain logreg probe.


# ── Helpers ─────────────────────────────────────────────────────────────
META_FIELDS = [
    "language", "cwe", "mutation", "category",
    "func_rate", "sec_rate", "func_secure_rate",
    "n_changed_new", "n_changed_old",
]


def load_embeddings(model: str, suffix: str, feature: str = "embeddings"):
    path = EMB_DIR / f"{model}_{suffix}.npz"
    data = np.load(path, allow_pickle=True)
    df = pd.DataFrame({k: data[k] for k in META_FIELDS if k in data.files})
    return data[feature], df


def load_npz(model: str, suffix: str, emb_dir: Path = None,
             metadata_overlay: Path = None, model_short: str = None):
    """Load a model's npz once. Returns the np.load object (lazy) plus the
    metadata DataFrame. Use `data[feature]` to slice individual feature
    arrays without re-reading the file.

    If `metadata_overlay` is given, the func_rate / sec_rate /
    func_secure_rate columns are replaced by the values in the overlay
    (joined on category × language × cwe × mutation). Lets us reuse a
    single big npz across multiple eval runs without copying it.
    """
    base = emb_dir if emb_dir is not None else EMB_DIR
    path = base / f"{model}_{suffix}.npz"
    data = np.load(path, allow_pickle=True)
    df = pd.DataFrame({k: data[k] for k in META_FIELDS if k in data.files})

    if metadata_overlay is not None:
        ov = pd.read_parquet(metadata_overlay)
        if model_short and "model" in ov.columns:
            ov = ov[ov["model"] == model_short]
        keys = ["category", "language", "cwe", "mutation"]
        ov = ov[keys + ["func_rate", "sec_rate", "func_secure_rate"]]
        before = len(df)
        df = df.merge(ov, on=keys, how="left", suffixes=("", "_ov"))
        n_overlapped = df["func_rate_ov"].notna().sum()
        for col in ("func_rate", "sec_rate", "func_secure_rate"):
            df[col] = df[f"{col}_ov"].combine_first(df[col]).astype(np.float32)
            df = df.drop(columns=[f"{col}_ov"])
        assert len(df) == before, (
            f"metadata overlay duplicated rows: {before} -> {len(df)} "
            f"(check that the overlay has no dup keys per (cat, lang, cwe, mutation))"
        )
        print(
            f"  metadata overlay applied: {n_overlapped} / {before} rows "
            f"updated from {metadata_overlay}",
            flush=True,
        )

    return data, df


def list_layer_features(data) -> list[str]:
    """Return all `layer_{idx}_{pool}` feature names present in the npz,
    sorted by layer index then pooling."""
    feats = [k for k in data.files if k.startswith("layer_") and (
        k.endswith("_last_pos") or k.endswith("_swap_pos")
    )]
    def sort_key(k):
        # layer_36_last_pos -> (36, 'last_pos')
        parts = k.split("_")
        return (int(parts[1]), "_".join(parts[2:]))
    return sorted(feats, key=sort_key)


def resolve_features(features: list[str] | None, data) -> list[str]:
    """Translate virtual feature names to concrete column names in this npz.

    Virtual aliases:
        layer_last_<pool>   → the last saved layer × <pool>, e.g.
                              for Qwen3 (layers up to 48): layer_last_last_pos
                              → layer_48_last_pos.
        layer_first_<pool>  → the first saved layer × <pool>.

    A None features list expands to all real layer_*_pos columns in the file
    (no virtual aliases — keeps existing default behaviour).
    """
    if features is None:
        return list_layer_features(data)
    if "layer_indices" not in data.files:
        return features  # nothing to resolve against
    layer_indices = sorted(int(x) for x in data["layer_indices"])
    last_idx = layer_indices[-1]
    first_idx = layer_indices[0]
    out = []
    for f in features:
        if f.startswith("layer_last_"):
            out.append(f"layer_{last_idx}_{f[len('layer_last_'):]}")
        elif f.startswith("layer_first_"):
            out.append(f"layer_{first_idx}_{f[len('layer_first_'):]}")
        else:
            out.append(f)
    return out


def load_significant_counts(path: Path) -> pd.DataFrame:
    """Load per-mutation Fisher significance flags and count significant mutations
    per (model, language, cwe). Aggregates across categories (a mutation is the
    same intervention regardless of category).

    Returns a DataFrame with columns:
        model, language, cwe, n_func_sig, n_sec_sig, n_either_sig
    """
    import json
    raw = json.loads(path.read_text())
    rows = []
    for k in raw["func"].keys():
        cat, lang, model, cwe, mut = k.split("|")
        rows.append((model, lang, cwe, mut, int(raw["func"][k]), int(raw["sec"][k])))
    df = pd.DataFrame(rows, columns=["model", "language", "cwe", "mutation",
                                      "func_sig", "sec_sig"])
    df["either_sig"] = ((df["func_sig"] == 1) | (df["sec_sig"] == 1)).astype(int)
    agg = df.groupby(["model", "language", "cwe"])[
        ["func_sig", "sec_sig", "either_sig"]
    ].sum().reset_index().rename(columns={
        "func_sig": "n_func_sig", "sec_sig": "n_sec_sig", "either_sig": "n_either_sig",
    })
    return agg


# Map target → significance count column (per-target filter).
TARGET_SIG_COL = {
    "func_rate": "n_func_sig",
    "sec_rate":  "n_sec_sig",
    "func_secure_rate": "n_either_sig",  # kept for back-compat with older runs
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression and (when y_true is binary) classification metrics.

    Always-present:
        mse, r2, mae         — regression error metrics (Brier-equivalent for
                               binary y; still meaningful summaries).

    Binary-only (NaN otherwise):
        auc                  — area under ROC.
        precision, recall, f1, accuracy — at threshold 0.5.

    The metric dict shape is constant across calls, so a CSV with all keys
    can be assembled regardless of whether the run is on temp-0 (binary) or
    temp-0.3 (continuous) outcomes.
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    var = float(np.var(y_true))
    r2 = 1.0 - mse / var if var > 0 else float("nan")

    uniq = np.unique(y_true)
    is_binary = (uniq.size <= 2) and set(uniq.tolist()).issubset({0.0, 1.0})
    if is_binary and uniq.size == 2:
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
        )
        y_true_int = y_true.astype(int)
        y_pred_class = (y_pred >= 0.5).astype(int)
        auc = (float(roc_auc_score(y_true_int, y_pred))
               if np.std(y_pred) > 0 else float("nan"))
        precision = float(precision_score(y_true_int, y_pred_class, zero_division=0))
        recall    = float(recall_score(y_true_int, y_pred_class, zero_division=0))
        f1        = float(f1_score(y_true_int, y_pred_class, zero_division=0))
        accuracy  = float(accuracy_score(y_true_int, y_pred_class))
    else:
        auc = precision = recall = f1 = accuracy = float("nan")

    return {
        "mse": mse, "r2": r2, "mae": mae,
        "auc": auc, "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
    }


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    hidden: tuple[int, ...], device: torch.device,
    dropout: float = None,
    weight_decay: float = None,
    return_train_pred: bool = False,
):
    """Train an MLP with early stopping on validation MSE.

    Returns the validation-fold predictions; if `return_train_pred=True`,
    returns `(train_pred, val_pred)`.
    """
    if dropout is None:
        dropout = MLP_DROPOUT
    if weight_decay is None:
        weight_decay = WEIGHT_DECAY
    X_tr_t  = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y_tr_t  = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    model = _MLP(X_tr.shape[1], hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True
    )

    best_mse = float("inf")
    best_pred = None
    best_train_pred = None
    no_improve = 0

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_val_t).cpu().numpy()
            if return_train_pred:
                train_pred = model(X_tr_t).cpu().numpy()
        mse = float(np.mean((y_val - pred) ** 2))
        if mse < best_mse - 1e-6:
            best_mse = mse
            best_pred = pred
            if return_train_pred:
                best_train_pred = train_pred
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    if return_train_pred:
        return best_train_pred, best_pred
    return best_pred


def run_cv(
    X: np.ndarray, y: np.ndarray, probe: str, device: torch.device,
    dropout: float = None,
    hidden: tuple[int, ...] = None,
    weight_decay: float = None,
    pca_dim: int = 0,
    logreg_c: float = None,
) -> dict:
    """5-fold CV. Returns per-fold metric arrays for the probe and the mean baseline.

    Baseline metrics use np.std(y_pred)==0, so Spearman ρ is NaN for the
    constant predictor by construction (no rank information). MSE and MAE are
    well-defined and worth reporting since they're directly comparable to the
    probe's MSE / MAE.
    """
    # Stratified folds for binary targets so each fold gets both classes when
    # possible — otherwise random KFold can hand us single-class test folds
    # (AUC undefined, F1 dominated by the majority). Plain KFold is fine for
    # continuous y where stratification isn't meaningful.
    uniq_y = np.unique(y)
    is_binary = (uniq_y.size == 2) and set(uniq_y.tolist()).issubset({0.0, 1.0})
    if is_binary:
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
        splits = list(cv.split(X, y.astype(int)))
    else:
        cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
        splits = list(cv.split(X))

    keys = ["mse", "r2", "mae",
            "auc", "precision", "recall", "f1", "accuracy"]
    probe_test_mtx  = {k: [] for k in keys}
    probe_train_mtx = {k: [] for k in keys}
    base_test_mtx   = {k: [] for k in keys}
    base_train_mtx  = {k: [] for k in keys}
    n_skipped = 0

    for tr_idx, te_idx in splits:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Skip folds with a single-class train or test set: the probe has
        # nothing to learn (train) or AUC is undefined (test). Stratified
        # splitting prevents this when the minority class has ≥ CV_SPLITS
        # samples; below that, some folds are unavoidably degenerate.
        if is_binary and (np.unique(y_tr).size < 2 or np.unique(y_te).size < 2):
            n_skipped += 1
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Optional PCA — fit on train fold only, then transform both.
        # Cap pca_dim at min(n_train-1, n_features) to avoid sklearn errors.
        if pca_dim and pca_dim > 0:
            from sklearn.decomposition import PCA
            n_comp = min(pca_dim, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
            if n_comp >= 1:
                pca = PCA(n_components=n_comp)
                X_tr_s = pca.fit_transform(X_tr_s)
                X_te_s = pca.transform(X_te_s)

        # Constant-mean baseline. Same predicted value (y_tr.mean()) on train
        # and test, so train-side baseline metrics simply describe variance
        # around that mean.
        base_pred_te = np.full_like(y_te, fill_value=y_tr.mean())
        base_pred_tr = np.full_like(y_tr, fill_value=y_tr.mean())
        for k, v in metrics(y_te, base_pred_te).items():
            base_test_mtx[k].append(v)
        for k, v in metrics(y_tr, base_pred_tr).items():
            base_train_mtx[k].append(v)

        if probe == "linreg":
            reg = Ridge(alpha=RIDGE_ALPHA)
            reg.fit(X_tr_s, y_tr)
            train_pred = reg.predict(X_tr_s)
            test_pred  = reg.predict(X_te_s)
        elif probe == "logreg":
            # Plain L2-regularised logistic regression with a fixed C. We
            # don't tune C via inner CV — there isn't enough data per CWE,
            # and the inner split crashes when the minority class is very
            # small. Re-run with a different --logreg-c to sweep manually.
            from sklearn.linear_model import LogisticRegression
            y_int = y_tr.astype(int)
            if np.unique(y_int).size < 2:
                p = float(y_int.mean())
                train_pred = np.full_like(y_tr, p)
                test_pred  = np.full_like(y_te, p)
            else:
                C = logreg_c if logreg_c is not None else LOGREG_C
                reg = LogisticRegression(C=C, max_iter=1000)
                reg.fit(X_tr_s, y_int)
                train_pred = reg.predict_proba(X_tr_s)[:, 1]
                test_pred  = reg.predict_proba(X_te_s)[:, 1]
        elif probe == "mlp1":
            h = hidden if hidden is not None else MLP1_HIDDEN
            train_pred, test_pred = fit_mlp(
                X_tr_s, y_tr, X_te_s, y_te, h, device,
                dropout=dropout, weight_decay=weight_decay, return_train_pred=True,
            )
        elif probe == "mlp2":
            h = hidden if hidden is not None else MLP2_HIDDEN
            train_pred, test_pred = fit_mlp(
                X_tr_s, y_tr, X_te_s, y_te, h, device,
                dropout=dropout, weight_decay=weight_decay, return_train_pred=True,
            )
        else:
            raise ValueError(f"Unknown probe: {probe}")

        for k, v in metrics(y_te, test_pred).items():
            probe_test_mtx[k].append(v)
        for k, v in metrics(y_tr, train_pred).items():
            probe_train_mtx[k].append(v)

    out = {f"probe_{k}":          np.array(v) for k, v in probe_test_mtx.items()}
    out.update({f"probe_train_{k}":    np.array(v) for k, v in probe_train_mtx.items()})
    out.update({f"baseline_{k}":       np.array(v) for k, v in base_test_mtx.items()})
    out.update({f"baseline_train_{k}": np.array(v) for k, v in base_train_mtx.items()})
    out["n_folds_used"]    = len(splits) - n_skipped
    out["n_folds_skipped"] = n_skipped
    return out


# ── Main ────────────────────────────────────────────────────────────────
def _parse_hidden(s: str) -> tuple[int, ...]:
    """`'512,256'` → (512, 256). `'default'` keeps the per-probe default."""
    if s == "default":
        return None
    return tuple(int(x) for x in s.split(","))


def main(suffix: str, probes: list[str], out_path: Path,
         features: list[str] | None, models: list[str],
         min_sig: int, min_sig_frac: float,
         dropouts: list[float], hidden_specs: list[str],
         weight_decays: list[float], pca_dims: list[int],
         emb_dir: Path = None, labels_path: Path = None,
         metadata_overlay: Path = None,
         logreg_cs: list[float] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    if emb_dir is None:
        emb_dir = EMB_DIR
    if labels_path is None:
        labels_path = STAT_LABELS_PATH
    print(f"Using emb_dir={emb_dir}, labels_path={labels_path}", flush=True)

    sig_counts = load_significant_counts(labels_path)
    print(f"Loaded significance counts ({len(sig_counts)} groups).")
    if min_sig > 0:
        print(f"  Filter active: ≥{min_sig} significant mutations.")
    if min_sig_frac > 0:
        print(f"  Filter active: ≥{min_sig_frac:.1%} significant fraction.")
    print("  Counts always saved in output (n_func_sig, n_sec_sig, n_either_sig, "
          "frac_*_sig) for post-hoc threshold sweeps.")

    # Build the cartesian-product of hyperparam configs.
    hp_configs = []
    for d in dropouts:
        for h_spec in hidden_specs:
            for wd in weight_decays:
                for pd_ in pca_dims:
                    hp_configs.append({
                        "dropout": d,
                        "hidden": _parse_hidden(h_spec),
                        "hidden_str": h_spec,
                        "weight_decay": wd,
                        "pca_dim": int(pd_),
                    })
    print(f"Hyperparam configs to sweep: {len(hp_configs)} "
          f"(dropouts × hidden_sizes × weight_decays × pca_dims = "
          f"{len(dropouts)} × {len(hidden_specs)} × {len(weight_decays)} × {len(pca_dims)})")

    metric_keys = ("mse", "r2", "mae",
                   "auc", "precision", "recall", "f1", "accuracy")

    rows = []
    for model in models:
        model_name = SHORT_NAMES[model]
        npz_path = emb_dir / f"{model}_{suffix}.npz"
        if not npz_path.exists():
            print(f"\n[SKIP] {model_name}: {npz_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"Model: {model_name}  ({npz_path.stat().st_size // 1_000_000} MB)")
        data, meta = load_npz(model, suffix, emb_dir=emb_dir,
                              metadata_overlay=metadata_overlay,
                              model_short=model)

        # Resolve which feature columns to probe (incl. layer_last_<pool> aliases).
        sweep_features = resolve_features(features, data)
        if not sweep_features:
            sweep_features = ["embeddings"] if "embeddings" in data.files else []
        if not sweep_features:
            print(f"  no probe-able features found; skipping")
            continue
        # Filter out features that don't exist in this npz (e.g. typo, or a
        # feature only present for some models).
        missing = [f for f in sweep_features if f not in data.files]
        if missing:
            print(f"  [WARN] features not in npz: {missing}")
            sweep_features = [f for f in sweep_features if f in data.files]
        print(f"  features to probe: {len(sweep_features)} "
              f"({sweep_features[0]} … {sweep_features[-1]})")

        keep_mask = (meta["mutation"] != "original").to_numpy()
        meta = meta[keep_mask].reset_index(drop=True)

        model_sig: dict[tuple[str, str], dict[str, int]] = {}
        sub = sig_counts[sig_counts["model"] == model]
        for _, r in sub.iterrows():
            model_sig[(r["language"], r["cwe"])] = {
                "n_func_sig": int(r["n_func_sig"]),
                "n_sec_sig": int(r["n_sec_sig"]),
                "n_either_sig": int(r["n_either_sig"]),
            }

        # Pre-compute per-(lang, cwe) masks once — reused across all features.
        groups: list[tuple[str, str, np.ndarray]] = []
        for lang in sorted(meta["language"].unique()):
            for cwe in sorted(meta[meta["language"] == lang]["cwe"].unique()):
                mask = ((meta["language"] == lang) & (meta["cwe"] == cwe)).to_numpy()
                if mask.sum() < MIN_SAMPLES:
                    continue
                groups.append((lang, cwe, mask))
        print(f"  {len(groups)} (lang, cwe) groups passing min_samples={MIN_SAMPLES}")

        for feature in sweep_features:
            print(f"\n  [feature] {feature}")
            X = data[feature]
            X = X[keep_mask]
            if X.dtype != np.float32:
                X = X.astype(np.float32)

            for lang, cwe, mask in groups:
                X_grp = X[mask]
                meta_grp = meta[mask]
                n_mut = int(mask.sum())
                sig = model_sig.get(
                    (lang, cwe),
                    {"n_func_sig": 0, "n_sec_sig": 0, "n_either_sig": 0},
                )

                for target_key, target_label in TARGETS:
                    n_sig = sig[TARGET_SIG_COL[target_key]]
                    frac_sig = n_sig / n_mut if n_mut > 0 else 0.0
                    if n_sig < min_sig or frac_sig < min_sig_frac:
                        continue

                    y = meta_grp[target_key].to_numpy(dtype=np.float32)

                    # Per-CWE class-count diagnostic (binary only). Same fold seed
                    # as run_cv so the printed counts match what each probe sees.
                    uniq_y = np.unique(y)
                    is_bin = (uniq_y.size == 2) and set(uniq_y.tolist()).issubset({0.0, 1.0})
                    pos_t_avg = neg_t_avg = pos_v_avg = neg_v_avg = float("nan")
                    if is_bin:
                        cv_diag = StratifiedKFold(
                            n_splits=CV_SPLITS, shuffle=True, random_state=42,
                        )
                        pos_t, neg_t, pos_v, neg_v = [], [], [], []
                        for tr, te in cv_diag.split(X_grp, y.astype(int)):
                            pos_t.append(int((y[tr] == 1).sum()))
                            neg_t.append(int((y[tr] == 0).sum()))
                            pos_v.append(int((y[te] == 1).sum()))
                            neg_v.append(int((y[te] == 0).sum()))
                        pos_t_avg = float(np.mean(pos_t))
                        neg_t_avg = float(np.mean(neg_t))
                        pos_v_avg = float(np.mean(pos_v))
                        neg_v_avg = float(np.mean(neg_v))
                        print(
                            f"    [{lang} {cwe} {target_label:11s}] n={n_mut}  "
                            f"train pos/neg = {pos_t_avg:5.1f}/{neg_t_avg:5.1f}  "
                            f"val pos/neg = {pos_v_avg:5.1f}/{neg_v_avg:5.1f}  "
                            f"frac_sig={frac_sig:.2f}",
                            flush=True,
                        )

                    sig_extras = {
                        "n_func_sig":     sig["n_func_sig"],
                        "n_sec_sig":      sig["n_sec_sig"],
                        "n_either_sig":   sig["n_either_sig"],
                        "frac_func_sig":   sig["n_func_sig"]   / n_mut,
                        "frac_sec_sig":    sig["n_sec_sig"]    / n_mut,
                        "frac_either_sig": sig["n_either_sig"] / n_mut,
                        # Average class counts across the 5 folds (binary targets).
                        "n_pos_train_avg": pos_t_avg,
                        "n_neg_train_avg": neg_t_avg,
                        "n_pos_val_avg":   pos_v_avg,
                        "n_neg_val_avg":   neg_v_avg,
                    }
                    for hp in hp_configs:
                        base_row = {
                            "model": model_name, "feature": feature,
                            "language": lang, "cwe": cwe,
                            "target": target_label,
                            "n": n_mut, "y_mean": float(y.mean()),
                            "y_var": float(np.var(y)),
                            "dropout": hp["dropout"],
                            "hidden": hp["hidden_str"],
                            "weight_decay": hp["weight_decay"],
                            "pca_dim": hp["pca_dim"],
                            **sig_extras,
                        }

                        # Per-probe sweep keys: logreg sweeps over `logreg_cs`,
                        # other probes don't consume that hyperparam.
                        cs_for = {p: (logreg_cs if p == "logreg" else [None])
                                  for p in probes}

                        if np.var(y) == 0:
                            for probe in probes:
                                for c in cs_for[probe]:
                                    row = {
                                        **base_row, "probe": probe, "note": "constant target",
                                        "n_folds_used": 0, "n_folds_skipped": CV_SPLITS,
                                        "logreg_c": float(c) if c is not None else float("nan"),
                                    }
                                    for prefix in ("", "train_", "baseline_", "baseline_train_"):
                                        for k in metric_keys:
                                            row[f"{prefix}{k}_mean"] = float("nan")
                                            row[f"{prefix}{k}_std"]  = float("nan")
                                    rows.append(row)
                            continue

                        for probe in probes:
                          for c in cs_for[probe]:
                            cv = run_cv(
                                X_grp, y, probe, device,
                                dropout=hp["dropout"],
                                hidden=hp["hidden"],
                                weight_decay=hp["weight_decay"],
                                pca_dim=hp["pca_dim"],
                                logreg_c=c,
                            )
                            note = ""
                            if cv["n_folds_used"] == 0:
                                note = "all folds degenerate"
                            elif cv["n_folds_skipped"] > 0:
                                note = f"{cv['n_folds_skipped']} fold(s) skipped (single-class)"
                            row = {
                                **base_row, "probe": probe, "note": note,
                                "n_folds_used":    cv["n_folds_used"],
                                "n_folds_skipped": cv["n_folds_skipped"],
                                "logreg_c": float(c) if c is not None else float("nan"),
                            }
                            for k in metric_keys:
                                arr_test    = cv[f"probe_{k}"]
                                arr_train   = cv[f"probe_train_{k}"]
                                arr_b_test  = cv[f"baseline_{k}"]
                                arr_b_train = cv[f"baseline_train_{k}"]
                                row[f"{k}_mean"]                = (float(np.nanmean(arr_test))    if arr_test.size    else float("nan"))
                                row[f"{k}_std"]                 = (float(np.nanstd(arr_test))     if arr_test.size    else float("nan"))
                                row[f"train_{k}_mean"]          = (float(np.nanmean(arr_train))   if arr_train.size   else float("nan"))
                                row[f"train_{k}_std"]           = (float(np.nanstd(arr_train))    if arr_train.size   else float("nan"))
                                row[f"baseline_{k}_mean"]       = (float(np.nanmean(arr_b_test))  if arr_b_test.size  else float("nan"))
                                row[f"baseline_{k}_std"]        = (float(np.nanstd(arr_b_test))   if arr_b_test.size  else float("nan"))
                                row[f"baseline_train_{k}_mean"] = (float(np.nanmean(arr_b_train)) if arr_b_train.size else float("nan"))
                                row[f"baseline_train_{k}_std"]  = (float(np.nanstd(arr_b_train))  if arr_b_train.size else float("nan"))
                            rows.append(row)
            # Per-feature summary. Picks the right metrics based on whether
            # this run is binary (classification) or continuous (regression).
            feat_rows = [r for r in rows if r["feature"] == feature
                         and r["model"] == model_name and r["note"] == ""]
            if feat_rows:
                # Binary if AUC is defined for at least some rows.
                aucs = [r["auc_mean"] for r in feat_rows
                        if not np.isnan(r["auc_mean"])]
                is_binary_run = len(aucs) > 0
                if is_binary_run:
                    f1_test  = np.nanmean([r["f1_mean"]       for r in feat_rows])
                    f1_train = np.nanmean([r["train_f1_mean"] for r in feat_rows])
                    auc_test = np.nanmean([r["auc_mean"]      for r in feat_rows])
                    acc_test = np.nanmean([r["accuracy_mean"] for r in feat_rows])
                    base_f1  = np.nanmean([r["baseline_f1_mean"]      for r in feat_rows])
                    base_acc = np.nanmean([r["baseline_accuracy_mean"] for r in feat_rows])
                    print(
                        f"    n_rows={len(feat_rows):4d}  "
                        f"⟨F1⟩ train→test: {f1_train:.3f}→{f1_test:.3f}  "
                        f"⟨AUC⟩={auc_test:.3f}  ⟨acc⟩={acc_test:.3f}  "
                        f"(baseline F1={base_f1:.3f}, acc={base_acc:.3f})"
                    )
                else:
                    rmse_p_test  = np.nanmean([np.sqrt(r["mse_mean"])          for r in feat_rows]) * 100
                    rmse_p_train = np.nanmean([np.sqrt(r["train_mse_mean"])    for r in feat_rows]) * 100
                    rmse_b_test  = np.nanmean([np.sqrt(r["baseline_mse_mean"]) for r in feat_rows]) * 100
                    print(
                        f"    n_rows={len(feat_rows):4d}  "
                        f"⟨RMSE pp⟩ probe train→test: {rmse_p_train:.1f}→{rmse_p_test:.1f}  "
                        f"(gap={rmse_p_test - rmse_p_train:+.1f})  "
                        f"baseline test={rmse_b_test:.1f}  "
                        f"(probe Δ={rmse_b_test - rmse_p_test:+.2f}pp)"
                    )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix", type=str, default="hidden_states_multilayer",
        help="Embedding file suffix (matches compute_hidden_states.py --out-suffix).",
    )
    parser.add_argument(
        "--probes", nargs="+", default=DEFAULT_PROBES, choices=PROBES,
        help="Which probes to run. Default: mlp2 only (Phase 1 sweep).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path. Default: embeddings/probe_results_<suffix>.csv",
    )
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Which feature arrays in the npz to probe. Default: all "
             "layer_{i}_{pool} columns found in the file.",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS, choices=MODELS,
        help="Which models to probe.",
    )
    parser.add_argument(
        "--min-sig", type=int, default=0,
        help="Minimum absolute number of FDR-significant mutations required.",
    )
    parser.add_argument(
        "--min-sig-frac", type=float, default=0.0,
        help="Minimum fraction (sig / n_mutations) of significant mutations required. "
             "Default 0 — no filter; rely on post-hoc filtering using the saved counts.",
    )
    parser.add_argument(
        "--dropouts", nargs="+", type=float, default=[MLP_DROPOUT],
        help="MLP dropout values to sweep. Default: [0.3].",
    )
    parser.add_argument(
        "--hidden-sizes", nargs="+", default=["default"],
        help="Comma-separated hidden layer widths to sweep, e.g. "
             "'512,256' '256,64'. 'default' keeps the per-probe default. "
             "Default: ['default'] = (512,256) for mlp2.",
    )
    parser.add_argument(
        "--weight-decays", nargs="+", type=float, default=[WEIGHT_DECAY],
        help="L2 weight-decay values to sweep. Default: [1e-4].",
    )
    parser.add_argument(
        "--pca-dims", nargs="+", type=int, default=[0],
        help="PCA dimensions to sweep (0 = no PCA). Fitted on train fold "
             "per outer CV split; transform applied to test fold.",
    )
    parser.add_argument(
        "--emb-dir", type=Path, default=None,
        help=f"Directory holding the multilayer npz files. Default: {EMB_DIR}",
    )
    parser.add_argument(
        "--labels-path", type=Path, default=None,
        help=f"Path to stat_change_labels.json. Default: {STAT_LABELS_PATH}",
    )
    parser.add_argument(
        "--logreg-c", nargs="+", type=float, default=[LOGREG_C],
        help=f"Inverse L2 strength(s) for the logreg probe — accepts a list "
             f"to sweep, e.g. '--logreg-c 0.001 0.01 0.1 1.0'. Default [{LOGREG_C}]. "
             "Smaller C = stronger regularization.",
    )
    parser.add_argument(
        "--metadata-overlay", type=Path, default=None,
        help="Optional parquet file with refreshed func_rate / sec_rate / "
             "func_secure_rate values, joined on (category, language, cwe, "
             "mutation). Lets us reuse a single hidden-states npz across "
             "multiple eval runs.",
    )
    args = parser.parse_args()
    if args.out is not None:
        out_path = args.out
    else:
        out_path = (args.emb_dir or EMB_DIR) / f"probe_results_{args.suffix}.csv"
    main(args.suffix, args.probes, out_path, args.features, args.models,
         args.min_sig, args.min_sig_frac,
         args.dropouts, args.hidden_sizes, args.weight_decays, args.pca_dims,
         emb_dir=args.emb_dir, labels_path=args.labels_path,
         metadata_overlay=args.metadata_overlay,
         logreg_cs=args.logreg_c)
