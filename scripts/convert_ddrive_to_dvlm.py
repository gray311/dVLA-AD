"""Convert Fast-dDrive weights to Fast-dVLM key naming so SGLang's
Fast-dVLM model class can load them. Saves to a new dir with dVLM's
config.json + modeling.py + tokenizer files.
"""
import glob, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

DDRIVE_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/transformers/models--xiwenyoumu--Fast-dDrive/snapshots/ddadfbbd31014fa0d6c3bbf457070d499ec19241"
DVLM_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dVLM_3B"
OUT_DIR = "/weka/home/ext-yingzima/scratchaszalay1_ssci/yy/huggingface/Fast_dDrive_as_dVLM"


def remap_key(k):
    """dDrive → dVLM naming.

    dDrive checkpoint keys:
      model.X (e.g. model.embed_tokens, model.layers.0.X)  → language layers
      visual.X                                              → vision tower

    dVLM expected keys:
      model.language_model.X  ← language layers
      model.visual.X           ← vision tower
    """
    if k.startswith("visual."):
        return "model." + k
    if k.startswith("model.") and not k.startswith("model.visual."):
        return "model.language_model." + k[len("model."):]
    return k


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Copy non-weight files from dVLM dir (config.json, modeling.py, etc.)
    print(f"Copying dVLM config/modeling/tokenizer → {OUT_DIR}")
    for fname in os.listdir(DVLM_DIR):
        src = os.path.join(DVLM_DIR, fname)
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(OUT_DIR, fname))

    # Load + remap + save dDrive weights with dVLM naming.
    print(f"\nLoading dDrive safetensors from {DDRIVE_DIR}")
    state_dict = {}
    for sf in sorted(glob.glob(f"{DDRIVE_DIR}/model-*.safetensors")):
        print(f"  {os.path.basename(sf)}")
        state_dict.update(load_file(sf))
    print(f"  total keys: {len(state_dict)}")

    remapped = {remap_key(k): v for k, v in state_dict.items()}
    print(f"  remapped keys: {len(remapped)}")
    # Sanity check
    sample_old = list(state_dict.keys())[:3]
    sample_new = [remap_key(k) for k in sample_old]
    for o, n in zip(sample_old, sample_new):
        print(f"    {o}  →  {n}")

    # Save as a single safetensors (or split into 2 like dVLM original).
    print(f"\nSaving to {OUT_DIR}/model.safetensors")
    # Split into 2 files to match dVLM's layout (~6GB each shouldn't be needed; just one is fine)
    out_path = os.path.join(OUT_DIR, "model.safetensors")
    save_file(remapped, out_path)
    # Index file for single shard
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in remapped.values())},
        "weight_map": {k: "model.safetensors" for k in remapped.keys()},
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
