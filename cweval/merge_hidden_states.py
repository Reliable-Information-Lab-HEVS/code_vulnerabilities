"""Merge per-job hidden-state staging files into the canonical npz.

`generate_local.py` writes one staging .npz per cweval.sh run under
`embeddings/staging/`. This script consolidates those into the schema
that `probe_embeddings.py` and downstream analysis already consume:

    embeddings/<model_short>_hidden_states_multilayer.npz

Schema (compatible with compute_hidden_states.py output):
    - category, language, cwe, mutation              (object arrays)
    - layer_indices                                  (int32 array)
    - layer_<idx>_last_pos                           (float16 2-D array, [N, hidden_dim])

If a row's `(category, language, cwe, mutation)` already exists in the
target npz, this script overwrites it with the new staging value
(`--keep=newest`) or skips it (`--keep=existing`). Default: newest wins.

Run after all concurrent cweval_local.sh jobs for a given model have
finished. Safe to re-run (idempotent up to ordering ties).

Example:
    python cweval/merge_hidden_states.py \
        --staging-dir embeddings/staging \
        --pattern 'Qwen3-Coder-30B-A3B-Instruct_*.npz' \
        --out embeddings/Qwen3-Coder-30B-A3B-Instruct_hidden_states_multilayer.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS = ('category', 'language', 'cwe', 'mutation')


def _load_npz_as_df(path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Return (metadata DataFrame, dict of layer_*_last_pos arrays)."""
    data = np.load(path, allow_pickle=True)
    meta = pd.DataFrame({c: data[c] for c in META_COLS if c in data.files})
    feats = {k: data[k] for k in data.files
             if k.startswith('layer_') and (k.endswith('_last_pos') or k.endswith('_swap_pos'))}
    layer_indices = data['layer_indices'] if 'layer_indices' in data.files else None
    return meta, feats, layer_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--staging-dir', type=Path, required=True,
                   help='Directory containing per-job staging npz files.')
    p.add_argument('--pattern', default='*.npz',
                   help='Glob pattern for staging files (default: *.npz).')
    p.add_argument('--out', type=Path, required=True,
                   help='Canonical merged npz output path.')
    p.add_argument('--keep', choices=['newest', 'existing'], default='newest',
                   help='On (cat,lang,cwe,mutation) collision: keep the newer '
                        'staging row (newest) or the row already in --out (existing). '
                        'Default: newest.')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    staging_files = sorted(args.staging_dir.glob(args.pattern))
    if not staging_files:
        print(f'No staging files matched {args.staging_dir / args.pattern}', flush=True)
        return

    # If --out exists, treat it as the first source so the existing rows
    # are preserved unless a staging file overrides them under `newest`.
    sources: list[Path] = []
    if args.out.exists():
        sources.append(args.out)
        print(f'  found existing {args.out} — merging into it', flush=True)
    sources.extend(staging_files)

    metas: list[pd.DataFrame] = []
    feats_by_key: dict[str, list[np.ndarray]] = {}
    layer_indices = None
    for src in sources:
        meta, feats, li = _load_npz_as_df(src)
        if not len(meta):
            continue
        meta = meta.copy()
        meta['_source'] = str(src)
        metas.append(meta)
        for k, arr in feats.items():
            feats_by_key.setdefault(k, []).append(arr)
        if li is not None:
            if layer_indices is None:
                layer_indices = li
            elif not np.array_equal(li, layer_indices):
                raise ValueError(
                    f'Layer indices mismatch between {src} ({li}) and earlier '
                    f'sources ({layer_indices}). Refusing to merge.'
                )
        print(f'  loaded {len(meta):6d} rows from {src.name}', flush=True)

    if not metas:
        print('  nothing to merge', flush=True)
        return

    all_meta = pd.concat(metas, ignore_index=True)
    # Concat the feature arrays in the same row order.
    feature_keys = sorted(feats_by_key)
    all_feats = {k: np.concatenate(feats_by_key[k], axis=0) for k in feature_keys}
    n_rows = len(all_meta)
    for k, arr in all_feats.items():
        assert len(arr) == n_rows, (
            f'Feature {k} has {len(arr)} rows but metadata has {n_rows}. '
            'Source files have inconsistent shapes.'
        )

    # Drop duplicates by (cat, lang, cwe, mutation). With --keep=newest the
    # later occurrence wins (staging files appear after the existing --out
    # in `sources`). With --keep=existing, the earlier occurrence wins.
    if args.keep == 'newest':
        keep_idx = all_meta.drop_duplicates(subset=list(META_COLS), keep='last').index
    else:
        keep_idx = all_meta.drop_duplicates(subset=list(META_COLS), keep='first').index
    n_dups = n_rows - len(keep_idx)

    final_meta = all_meta.loc[keep_idx].reset_index(drop=True)
    keep_mask = np.zeros(n_rows, dtype=bool)
    keep_mask[keep_idx] = True
    final_feats = {k: arr[keep_mask] for k, arr in all_feats.items()}

    print(f'  total rows merged: {n_rows}, kept: {len(final_meta)}, '
          f'dropped duplicates: {n_dups}', flush=True)

    if args.dry_run:
        print(f'  --dry-run: would write {args.out}', flush=True)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save = {
        'category': final_meta['category'].to_numpy(),
        'language': final_meta['language'].to_numpy(),
        'cwe':      final_meta['cwe'].to_numpy(),
        'mutation': final_meta['mutation'].to_numpy(),
    }
    if layer_indices is not None:
        save['layer_indices'] = layer_indices
    save.update(final_feats)
    np.savez_compressed(args.out, **save)
    print(f'  wrote {args.out} ({len(final_meta)} rows, '
          f'{len(feature_keys)} layer features)', flush=True)


if __name__ == '__main__':
    main()
