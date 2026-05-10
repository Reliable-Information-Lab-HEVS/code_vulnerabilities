"""Compute embeddings of only the changed token(s) for each mutation.

Same overall pipeline as compute_embeddings.py, but for each mutation we
tokenize the original and mutated prompts, find the differing positions
(longest common prefix/suffix strip), and mean-pool only the new tokens'
embeddings (instead of the whole prompt).

Output mirrors compute_embeddings.py so probe_embeddings.py works unchanged
with --suffix changed_token.

Saved arrays:
    embeddings:        (N, D)  mean of new-token embeddings   ← used by probe
    embeddings_delta:  (N, D)  mean(new) - mean(old)          ← for comparison
    n_changed_new, n_changed_old
    language, cwe, mutation, category
    func_rate, sec_rate, func_secure_rate
"""

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from compute_embeddings import (
    HF_NAMES, LANGUAGES, INITIAL_CATEGORIES, LANG_EXT,
    BASE_DATASETS, BASE_EVALS, OUT_DIR,
    _FNAME_TASK,
    find_prompt, load_embedding_matrix, _strip_lang_suffix,
    load_results,
)


def find_changed_token_ids(orig_ids: list[int], mut_ids: list[int]):
    """Return (new_changed, old_changed) by stripping common prefix and suffix."""
    p = 0
    n_min = min(len(orig_ids), len(mut_ids))
    while p < n_min and orig_ids[p] == mut_ids[p]:
        p += 1
    s = 0
    while (
        s < min(len(orig_ids) - p, len(mut_ids) - p)
        and orig_ids[len(orig_ids) - 1 - s] == mut_ids[len(mut_ids) - 1 - s]
    ):
        s += 1
    new = mut_ids[p : len(mut_ids) - s]
    old = orig_ids[p : len(orig_ids) - s]
    return new, old


def process_model(
    model_short: str,
    device: torch.device,
    categories: list[str],
    evals_dir: Path,
    out_path: Path,
):
    hf_name = HF_NAMES[model_short]
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {model_short} ({hf_name})", flush=True)
    print(f"  categories: {categories}", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    emb_matrix = load_embedding_matrix(hf_name).to(device)
    emb_dim = emb_matrix.shape[1]
    print(f"  Embedding dim: {emb_dim}, vocab size: {emb_matrix.shape[0]}", flush=True)

    all_emb_new, all_emb_delta = [], []
    meta = {k: [] for k in [
        "language", "cwe", "mutation", "category",
        "func_rate", "sec_rate", "func_secure_rate",
        "n_changed_new", "n_changed_old",
    ]}
    n_no_diff = 0
    n_only_delete = 0

    for category in categories:
        for language in LANGUAGES:
            print(f"\n  [{category}] Language: {language}", flush=True)
            dataset_dir = BASE_DATASETS / f"{language}_{category}_{model_short}"
            evals_subdir = evals_dir / f"{language}_{category}_{model_short}"
            if not dataset_dir.exists():
                print(f"    SKIP: {dataset_dir} not found", flush=True)
                continue
            if not (evals_subdir / "res_all.json").exists():
                print(f"    SKIP: {evals_subdir / 'res_all.json'} not found", flush=True)
                continue

            results = load_results(evals_dir, language, category, model_short)
            ext = LANG_EXT[language]
            task_pat = re.compile(_FNAME_TASK[category].format(ext=ext))

            # Tokenize originals once per cwe.
            originals: dict[str, list[int]] = {}
            mutation_paths: list[tuple[Path, str, str]] = []
            for tf in sorted(dataset_dir.glob(f"*_task.{ext}")):
                fname = tf.name
                if "mutated" in fname:
                    m = task_pat.match(fname)
                    if not m:
                        continue
                    mutation_paths.append((tf, _strip_lang_suffix(m["cwe"]), m["mut"]))
                else:
                    cwe_clean = _strip_lang_suffix(fname.replace(f"_task.{ext}", ""))
                    try:
                        prompt = find_prompt(tf.read_text(), language)
                    except ValueError:
                        continue
                    originals[cwe_clean] = tokenizer.encode(
                        prompt, add_special_tokens=False
                    )

            # Build per-mutation (new_ids, old_ids) lists, paired with metadata.
            batch_new_ids: list[list[int]] = []
            batch_old_ids: list[list[int]] = []
            batch_meta: list[tuple[str, str, str, str, dict]] = []
            skipped = 0
            for tf, cwe_clean, mut_key in mutation_paths:
                res = results.get((cwe_clean, mut_key))
                if res is None or cwe_clean not in originals:
                    skipped += 1
                    continue
                try:
                    prompt = find_prompt(tf.read_text(), language)
                except ValueError:
                    skipped += 1
                    continue
                mut_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if not mut_ids:
                    skipped += 1
                    continue
                new_ids, old_ids = find_changed_token_ids(
                    originals[cwe_clean], mut_ids
                )
                if not new_ids and not old_ids:
                    n_no_diff += 1
                    skipped += 1
                    continue
                if not new_ids:
                    n_only_delete += 1
                    skipped += 1
                    continue
                batch_new_ids.append(new_ids)
                batch_old_ids.append(old_ids)
                batch_meta.append((language, cwe_clean, mut_key, category, res))

            # Batched GPU lookup, mirroring compute_embeddings.py.
            BATCH_SIZE = 512
            for i in range(0, len(batch_new_ids), BATCH_SIZE):
                chunk_new = batch_new_ids[i : i + BATCH_SIZE]
                chunk_old = batch_old_ids[i : i + BATCH_SIZE]
                chunk_meta = batch_meta[i : i + BATCH_SIZE]

                flat_new = [t for ids in chunk_new for t in ids]
                lens_new = [len(ids) for ids in chunk_new]
                flat_old = [t for ids in chunk_old for t in ids]
                lens_old = [len(ids) for ids in chunk_old]

                new_embs = emb_matrix[torch.tensor(flat_new, device=device)]
                old_embs = (
                    emb_matrix[torch.tensor(flat_old, device=device)]
                    if flat_old else None
                )

                off_n = off_o = 0
                for ln, lo, (lang_val, cwe_val, mut_val, cat_val, res_val) in zip(
                    lens_new, lens_old, chunk_meta
                ):
                    mean_new = new_embs[off_n : off_n + ln].mean(dim=0).cpu().numpy()
                    off_n += ln
                    if lo > 0:
                        mean_old = old_embs[off_o : off_o + lo].mean(dim=0).cpu().numpy()
                        off_o += lo
                    else:
                        mean_old = np.zeros_like(mean_new)

                    all_emb_new.append(mean_new)
                    all_emb_delta.append(mean_new - mean_old)
                    meta["language"].append(lang_val)
                    meta["cwe"].append(cwe_val)
                    meta["mutation"].append(mut_val)
                    meta["category"].append(cat_val)
                    meta["func_rate"].append(res_val["func_rate"])
                    meta["sec_rate"].append(res_val["sec_rate"])
                    meta["func_secure_rate"].append(res_val["func_secure_rate"])
                    meta["n_changed_new"].append(ln)
                    meta["n_changed_old"].append(lo)

            print(f"    Processed: {len(batch_meta)}, Skipped: {skipped}", flush=True)

    print(
        f"\n  no-diff after tokenization: {n_no_diff}, pure-delete: {n_only_delete}",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=np.array(all_emb_new, dtype=np.float32),
        embeddings_delta=np.array(all_emb_delta, dtype=np.float32),
        language=np.array(meta["language"]),
        cwe=np.array(meta["cwe"]),
        mutation=np.array(meta["mutation"]),
        category=np.array(meta["category"]),
        func_rate=np.array(meta["func_rate"], dtype=np.float32),
        sec_rate=np.array(meta["sec_rate"], dtype=np.float32),
        func_secure_rate=np.array(meta["func_secure_rate"], dtype=np.float32),
        n_changed_new=np.array(meta["n_changed_new"], dtype=np.int32),
        n_changed_old=np.array(meta["n_changed_old"], dtype=np.int32),
    )
    print(f"\n  Saved {len(all_emb_new)} embeddings to {out_path}", flush=True)

    del emb_matrix
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=list(HF_NAMES.keys()),
        choices=list(HF_NAMES.keys()),
    )
    parser.add_argument(
        "--categories", nargs="+",
        default=INITIAL_CATEGORIES,
    )
    parser.add_argument("--evals-dir", type=Path, default=BASE_EVALS)
    parser.add_argument("--out-suffix", type=str, default="changed_token")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_short in args.models:
        out_path = OUT_DIR / f"{model_short}_{args.out_suffix}.npz"
        process_model(model_short, device, args.categories, args.evals_dir, out_path)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
