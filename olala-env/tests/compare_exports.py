"""Compare a fresh mg2hf export against the reference export of the same iteration."""
import json, sys, hashlib, torch
from safetensors import safe_open
new, ref = sys.argv[1], sys.argv[2]
cn, cr = json.load(open(f"{new}/config.json")), json.load(open(f"{ref}/config.json"))
only_new = {k: cn[k] for k in cn if k not in cr}; only_ref = {k: cr[k] for k in cr if k not in cn}
diff = {k: (cr[k], cn[k]) for k in cn if k in cr and cn[k] != cr[k]}
print("config: keys only in new:", only_new); print("config: keys only in ref:", only_ref); print("config: differing values:", diff)
def tensors(d):
    out = {}
    import glob
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys(): out[k] = s.get_tensor(k)
    return out
tn, tr = tensors(new), tensors(ref)
print(f"tensors: new={len(tn)} ref={len(tr)} | only new: {sorted(set(tn)-set(tr))[:5]} | only ref: {sorted(set(tr)-set(tn))[:5]}")
same = shape_mismatch = value_mismatch = 0; worst = []
for k in sorted(set(tn) & set(tr)):
    a, b = tn[k], tr[k]
    if a.shape != b.shape and a.numel() != b.numel(): shape_mismatch += 1; worst.append((k, tuple(a.shape), tuple(b.shape))); continue
    if a.dtype != b.dtype: worst.append((k, str(a.dtype), str(b.dtype)))
    if torch.equal(a.reshape(-1), b.reshape(-1)): same += 1
    else:
        value_mismatch += 1
        if len(worst) < 8: worst.append((k, (a.float()-b.float()).abs().max().item()))
print(f"bit-identical: {same} | value mismatch: {value_mismatch} | shape mismatch: {shape_mismatch}")
for w in worst[:8]: print("  ", w)
def h(p): return hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
for f in ("modeling_olala.py", "configuration_olala.py", "tokenizer.json", "chat_template.jinja", "tokenizer_config.json"):
    try: print(f"{f:26} new {h(f'{new}/{f}')}  ref {h(f'{ref}/{f}')}  {'same' if h(f'{new}/{f}')==h(f'{ref}/{f}') else 'DIFFERENT'}")
    except FileNotFoundError as e: print(f"{f:26} missing: {e.filename}")
print("MAIN modeling:", h(sys.argv[3]) if len(sys.argv) > 3 else "-")
