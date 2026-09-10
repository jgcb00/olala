"""verl-style packed row: two documents in one (1, T) row with position_ids
resetting at the second document, no attention_mask. With document boundaries
honoured, each document's logits must match its own single-sequence forward."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ckpt = sys.argv[1]; dev = "cuda"
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
docs = ["The capital of France is Paris. It is known for the Eiffel Tower and", "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) +"]
ids = [tok(d, return_tensors="pt").input_ids.to(dev) for d in docs]
with torch.no_grad():
    sep = [m(i).logits[0].float() for i in ids]
    packed = torch.cat(ids, dim=1); pos = torch.cat([torch.arange(i.shape[1]) for i in ids])[None].to(dev)
    out = m(input_ids=packed, position_ids=pos).logits[0].float()
    out_nopos = m(input_ids=packed).logits[0].float()   # no resets -> one document (control)
    # noise floor: the SAME document made longer. Causality says positions
    # 0..a-1 cannot change, so whatever does change is bf16 kernel numerics
    # that depend on the row length (largest at the first token, where the
    # massive activations live). Measured 2026-09-10 on 7A1B: 8.6 logits at
    # position 0, ~1.3 elsewhere, 88% top-1 -- identical to the packed case.
    ext = tok(docs[0] + " the Louvre museum, which houses the Mona Lisa and many other famous works", return_tensors="pt").input_ids.to(dev)
    floor = m(ext).logits[0].float()
a, b = ids[0].shape[1], ids[1].shape[1]
def cmp(x, y): return (x - y).abs().max().item(), (x.argmax(-1) == y.argmax(-1)).float().mean().item()
f0 = cmp(floor[:a], sep[0]); d0 = cmp(out[:a], sep[0]); d1 = cmp(out[a:], sep[1]); c1 = cmp(out_nopos[a:], sep[1])
print(f"noise floor (doc0 alone vs doc0 extended with its own text): max|dlogit|={f0[0]:.3f} top1 agree={f0[1]*100:.0f}%")
print(f"doc0 (packed vs alone): max|dlogit|={d0[0]:.3f} top1 agree={d0[1]*100:.0f}%")
print(f"doc1 (packed vs alone): max|dlogit|={d1[0]:.3f} top1 agree={d1[1]*100:.0f}%")
print(f"control, doc1 with NO position resets: max|dlogit|={c1[0]:.3f} top1 agree={c1[1]*100:.0f}%  (must be clearly worse)")
ok = d0[1] >= f0[1] - 0.05 and d1[1] >= f0[1] - 0.05 and c1[1] < min(d0[1], d1[1]) - 0.05
print("PACKED BOUNDARIES OK" if ok else "PACKED BOUNDARIES SUSPICIOUS")
