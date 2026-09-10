"""HF-side check of jgcb00/olala main: load a raw export with the NEW modeling
(no artificial_seq_len), prefill+decode on GPU, batched packed prefill."""
import os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ckpt = sys.argv[1]; dev = "cuda"
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
print(f"loaded {type(m).__name__} from {type(m).__module__} in {time.time()-t0:.0f}s; "
      f"has artificial_seq_len attr on config: {hasattr(m.config, 'artificial_seq_len')}")
assert not hasattr(m.config, "artificial_seq_len") or m.config.artificial_seq_len in (None, 0)
zero_dim = [n for n, p in m.named_parameters() if p.dim() == 0]; assert not zero_dim, zero_dim[:3]
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
with torch.no_grad():
    out = m.generate(ids, max_new_tokens=48, do_sample=False)
text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
print("GEN:", repr(text[:300]))
# decode-path consistency: logits of a full forward vs step-wise cache must agree on the generated tokens
with torch.no_grad():
    full = m(out).logits[0, ids.shape[1]-1:-1].float()
    step_pred = full.argmax(-1)
    agree = (step_pred == out[0, ids.shape[1]:]).float().mean().item()
print(f"greedy tokens reproduced by a single full forward: {agree*100:.0f}%")
# packed / padded batch prefill agrees with single-sequence forward
p2 = tok.apply_chat_template([{"role": "user", "content": "Name three primary colours."}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
tok.padding_side = "left"
batch = tok.pad({"input_ids": [ids[0].tolist(), p2[0].tolist()]}, return_tensors="pt").to(dev)
with torch.no_grad():
    lb = m(**batch).logits.float()
    l1 = m(ids).logits[0, -1].float(); l2 = m(p2).logits[0, -1].float()
d1 = (lb[0, -1] - l1).abs().max().item(); d2 = (lb[1, -1] - l2).abs().max().item()
print(f"batched vs single last-token logits max|diff|: {d1:.3f} {d2:.3f}; top1 agree: {lb[0,-1].argmax()==l1.argmax()} {lb[1,-1].argmax()==l2.argmax()}")
print("HF CHECK OK" if agree > 0.9 and lb[0,-1].argmax()==l1.argmax() and lb[1,-1].argmax()==l2.argmax() else "HF CHECK SUSPICIOUS")
