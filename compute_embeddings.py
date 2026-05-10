"""Compute prompt embeddings using each model's own embedding matrix.

For each (model, language, category) triple, extracts the prompt text from every
task file (original + mutated), tokenizes with the model's tokenizer, mean-pools
the token embeddings, and saves the result alongside outcome rates.

By default, processes all three initial mutation categories together and writes
one combined `.npz` per model to embeddings/<model_short>_initial.npz.

Output per model: embeddings/<model_short>_<suffix>.npz containing:
    embeddings:        np.ndarray (N, D)
    language, cwe, mutation, category: 1-D string arrays
    func_rate, sec_rate, func_secure_rate: 1-D float arrays in [0, 1]
"""

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────
HF_NAMES = {
    "CodeLlama-70b-Instruct-hf": "codellama/CodeLlama-70b-Instruct-hf",
    "deepseek-coder-33b-instruct": "deepseek-ai/deepseek-coder-33b-instruct",
    "Qwen3-Coder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}
LANGUAGES = ["c", "cpp", "go", "js", "py"]
INITIAL_CATEGORIES = [
    "mutated_prompts_character",
    "mutated3_prompts_character",
    "mutated_token_replacement",
]
LANG_EXT = {"c": "c", "cpp": "cpp", "go": "go", "js": "js", "py": "py"}

BASE_DATASETS = Path("/cluster/raid/home/stea/CWEval/new_datasets")
BASE_EVALS = Path("/cluster/raid/home/stea/CWEval/evals_temp03")
OUT_DIR = Path("/cluster/raid/home/stea/CWEval/embeddings")

# Filename patterns. Capture the CWE id and the mutation key.
#   token replacement:    <cwe>_<lang>_mutated_token_<pos>_task.<ext>
#   character mutations:  <cwe>_<lang>_mutated_<pos>_<variant>_task.<ext>
_FNAME_RES = {  # eval JSON entries end in _test.<py> for all categories
    "mutated_token_replacement":   re.compile(r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_token_(?P<mut>\d+)_test\.[a-z]+$"),
    "mutated_prompts_character":   re.compile(r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_(?P<mut>\d+_\d+)_test\.[a-z]+$"),
    "mutated3_prompts_character":  re.compile(r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_(?P<mut>\d+_\d+)_test\.[a-z]+$"),
}
_FNAME_TASK = {
    "mutated_token_replacement":   r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_token_(?P<mut>\d+)_task\.{ext}$",
    "mutated_prompts_character":   r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_(?P<mut>\d+_\d+)_task\.{ext}$",
    "mutated3_prompts_character":  r"(?P<cwe>.+?)(?:_[a-z]+)?_mutated_(?P<mut>\d+_\d+)_task\.{ext}$",
}


# ── Prompt extraction (mirrors generate_token_replacement.py) ───────────
def find_prompt(content: str, language: str) -> str:
    if language == "py":
        for delim in ["'''", '"""']:
            start = content.find(delim)
            if start != -1:
                end = content.find(delim, start + 3)
                if end != -1:
                    return content[start + 3 : end].strip()
        raise ValueError("No prompt delimiters found")
    else:
        bp = content.find("BEGIN PROMPT")
        bs = content.find("BEGIN SOLUTION")
        if bp == -1 or bs == -1:
            raise ValueError("No BEGIN PROMPT/BEGIN SOLUTION found")
        section = content[bp:bs]
        start = section.find("/**")
        offset = 3
        if start == -1:
            start = section.find("/*")
            offset = 2
        end = section.find("*/")
        if start == -1 or end == -1:
            raise ValueError("No comment delimiters found")
        return section[start + offset : end].strip()


# ── Embedding matrix loading ────────────────────────────────────────────
def load_embedding_matrix(model_name: str) -> torch.Tensor:
    from huggingface_hub import hf_hub_download

    try:
        from safetensors.torch import load_file

        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        for key, shard in index["weight_map"].items():
            if "embed_tokens" in key:
                shard_path = hf_hub_download(model_name, shard)
                tensors = load_file(shard_path)
                print(f"  Loaded embedding from {shard} ({key})")
                return tensors[key].float()
    except Exception as e:
        print(f"  Safetensors index failed ({e}), trying single file...")

    try:
        from safetensors.torch import load_file

        path = hf_hub_download(model_name, "model.safetensors")
        tensors = load_file(path)
        for key in tensors:
            if "embed_tokens" in key:
                print(f"  Loaded from model.safetensors ({key})")
                return tensors[key].float()
    except Exception as e:
        print(f"  Single safetensors failed ({e}), loading full model...")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    weights = model.get_input_embeddings().weight.detach().float()
    del model
    gc.collect()
    return weights


def _strip_lang_suffix(cwe_code: str) -> str:
    for lang_suffix in ["_c", "_cpp", "_go", "_js", "_py"]:
        if cwe_code.endswith(lang_suffix):
            return cwe_code[: -len(lang_suffix)]
    return cwe_code


# ── Result loading ──────────────────────────────────────────────────────
def load_results(evals_dir: Path, language: str, category: str, model_short: str) -> dict:
    """Load res_all.json and index by (cwe, mutation_key) for one category."""
    name = f"{language}_{category}_{model_short}"
    path = evals_dir / name / "res_all.json"
    raw = json.load(open(path))
    pat = _FNAME_RES[category]

    indexed = {}
    for key, vals in raw.items():
        ident = key.split("/")[-1]

        if "mutated" in ident:
            m = pat.match(ident)
            if not m:
                continue
            cwe_code = _strip_lang_suffix(m["cwe"])
            mut_key = m["mut"]
        else:
            cwe_code = _strip_lang_suffix(ident.replace("_test.py", ""))
            mut_key = "original"

        n = len(vals["functional"])
        func_rate = sum(vals["functional"]) / n
        sec_rate = sum(vals["secure"]) / n
        # Prefer the eval-provided func_secure list (functional AND secure);
        # fall back to elementwise AND if it is missing.
        if "func_secure" in vals:
            func_secure_rate = sum(vals["func_secure"]) / n
        else:
            func_secure_rate = sum(
                f and s for f, s in zip(vals["functional"], vals["secure"])
            ) / n
        indexed[(cwe_code, mut_key)] = {
            "func_rate": func_rate,
            "sec_rate": sec_rate,
            "func_secure_rate": func_secure_rate,
        }
    return indexed


# ── Main ────────────────────────────────────────────────────────────────
def process_model(
    model_short: str,
    device: torch.device,
    categories: list[str],
    evals_dir: Path,
    out_path: Path,
):
    hf_name = HF_NAMES[model_short]
    print(f"\n{'='*60}")
    print(f"Processing {model_short} ({hf_name})")
    print(f"  categories: {categories}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    emb_matrix = load_embedding_matrix(hf_name).to(device)
    emb_dim = emb_matrix.shape[1]
    print(f"  Embedding dim: {emb_dim}, vocab size: {emb_matrix.shape[0]}")

    all_embeddings = []
    meta = {k: [] for k in [
        "language", "cwe", "mutation", "category",
        "func_rate", "sec_rate", "func_secure_rate",
    ]}

    for category in categories:
        for language in LANGUAGES:
            print(f"\n  [{category}] Language: {language}")
            dataset_dir = BASE_DATASETS / f"{language}_{category}_{model_short}"
            evals_subdir = evals_dir / f"{language}_{category}_{model_short}"
            if not dataset_dir.exists():
                print(f"    SKIP: {dataset_dir} not found")
                continue
            if not (evals_subdir / "res_all.json").exists():
                print(f"    SKIP: {evals_subdir / 'res_all.json'} not found")
                continue

            results = load_results(evals_dir, language, category, model_short)
            ext = LANG_EXT[language]
            task_pat = re.compile(_FNAME_TASK[category].format(ext=ext))

            task_files = sorted(dataset_dir.glob(f"*_task.{ext}"))
            skipped = 0

            batch_token_ids = []
            batch_meta = []
            for tf in task_files:
                fname = tf.name
                if "mutated" in fname:
                    m = task_pat.match(fname)
                    if not m:
                        skipped += 1
                        continue
                    cwe_clean = _strip_lang_suffix(m["cwe"])
                    mut_key = m["mut"]
                else:
                    cwe_clean = _strip_lang_suffix(fname.replace(f"_task.{ext}", ""))
                    mut_key = "original"

                res = results.get((cwe_clean, mut_key))
                if res is None:
                    skipped += 1
                    continue

                try:
                    prompt = find_prompt(tf.read_text(), language)
                except ValueError:
                    skipped += 1
                    continue

                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if not token_ids:
                    skipped += 1
                    continue

                batch_token_ids.append(token_ids)
                batch_meta.append((language, cwe_clean, mut_key, category, res))

            BATCH_SIZE = 512
            for i in range(0, len(batch_token_ids), BATCH_SIZE):
                chunk_ids = batch_token_ids[i : i + BATCH_SIZE]
                chunk_meta = batch_meta[i : i + BATCH_SIZE]

                flat_ids = [tid for ids in chunk_ids for tid in ids]
                lengths = [len(ids) for ids in chunk_ids]

                flat_tensor = torch.tensor(flat_ids, device=device)
                flat_embs = emb_matrix[flat_tensor]

                offset = 0
                for length, (lang_val, cwe_val, mut_val, cat_val, res_val) in zip(
                    lengths, chunk_meta
                ):
                    mean_emb = flat_embs[offset : offset + length].mean(dim=0).cpu().numpy()
                    offset += length

                    all_embeddings.append(mean_emb)
                    meta["language"].append(lang_val)
                    meta["cwe"].append(cwe_val)
                    meta["mutation"].append(mut_val)
                    meta["category"].append(cat_val)
                    meta["func_rate"].append(res_val["func_rate"])
                    meta["sec_rate"].append(res_val["sec_rate"])
                    meta["func_secure_rate"].append(res_val["func_secure_rate"])

            print(f"    Processed: {len(batch_meta)}, Skipped: {skipped}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=np.array(all_embeddings, dtype=np.float32),
        language=np.array(meta["language"]),
        cwe=np.array(meta["cwe"]),
        mutation=np.array(meta["mutation"]),
        category=np.array(meta["category"]),
        func_rate=np.array(meta["func_rate"], dtype=np.float32),
        sec_rate=np.array(meta["sec_rate"], dtype=np.float32),
        func_secure_rate=np.array(meta["func_secure_rate"], dtype=np.float32),
    )
    print(f"\n  Saved {len(all_embeddings)} embeddings to {out_path}")

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
        help="Mutation categories to combine (default: the three initial categories).",
    )
    parser.add_argument(
        "--evals-dir", type=Path, default=BASE_EVALS,
        help=f"Eval results dir (default: {BASE_EVALS}).",
    )
    parser.add_argument(
        "--out-suffix", type=str, default="initial",
        help="Suffix for the output npz: <model>_<suffix>.npz (default: 'initial').",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_short in args.models:
        out_path = OUT_DIR / f"{model_short}_{args.out_suffix}.npz"
        process_model(model_short, device, args.categories, args.evals_dir, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
