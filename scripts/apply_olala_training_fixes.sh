#!/usr/bin/env bash
# apply_olala_training_fixes.sh — patch the third-party packages that still need
# it, in an EXISTING environment. Idempotent.
#
# Deliberately short. Everything that could be fixed at its source has been, so
# this only touches packages we do not control:
#
#   1. scattermoe parallel_experts.py : per-module dtype cast of inputs/gates to
#      the expert weight dtype, forward and backward (FSDP bf16 mixed
#      precision). Not on PyPI, pinned by commit; upstream has no fix.
#   2. transfer_queue simple_storage.py + controller.py : one malformed ZMQ
#      frame must not kill a storage worker thread permanently.
#   3. verl vllm_rollout.py + utils.py : the colocate weight-sync ZMQ socket
#      hardcodes a shared /tmp path, so a stale socket left by ANOTHER user
#      blocks every new run with "Address already in use". Use TMPDIR on both
#      sides.
#
# NOT here any more, because the need is gone rather than moved:
#   - vllm olala/mamba3.py + models/olala.py : both fixes are committed in the
#     vllm fork at the pinned ref. Nothing to patch.
#   - mamba mamba3_mimo.py + mamba3_siso_combined.py : the saved_tensors fix is
#     upstream in the pinned mamba ref (761b409, "single-unpack").
#   - checkpoint modeling_olala.py : the converter (../convert/) builds the HF
#     model from THIS repo's modeling and save_pretrained() ships it, so an
#     export already carries every fix. Re-export instead of patching.
#   - checkpoint 0-dim scale/bias weights : modeling_olala.py widens them in
#     memory after loading and writes them back 0-dim on save.
#
# Usage:
#   ./apply_olala_training_fixes.sh \
#       --python     /path/to/venv/bin/python \
#       --scattermoe /path/to/scattermoe/clone
#
# --scattermoe may be omitted; that fix is then skipped. transfer_queue and verl
# are located through the given python's installed packages. Every modified file
# gets a one-time .bak-olala-fixes backup next to it.
set -euo pipefail

PYBIN=python3 SCATTER=""
while [[ $# -gt 0 ]]; do case $1 in
  --python)     PYBIN=$2;   shift 2;;
  --scattermoe) SCATTER=$2; shift 2;;
  -h|--help) tail -n +2 "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
  *) echo "unknown arg: $1 (see --help)"; exit 1;;
esac; done

export OLALA_FIX_SCATTER="$SCATTER"
exec "$PYBIN" - <<'PY'
import importlib.util, os, shutil, sys
from pathlib import Path

results = []

def patch(name, path, fn, required=False):
    """required=True: a missing target is a FAIL, not a SKIP.

    Use it wherever the file MUST be there once its package is installed. A
    silent SKIP there means a rename upstream quietly disables the fix while
    the applier still exits 0 -- exactly how the dragon->olala rename turned
    off both vllm fixes unnoticed.
    """
    p = Path(path) if path else None
    if not p or not p.exists():
        status = "FAIL" if required else "SKIP"
        results.append((name, status, f"not found: {path or '(no path given)'}"))
        return
    text = p.read_text()
    status, new = fn(text)
    if status == "apply":
        bak = Path(str(p) + ".bak-olala-fixes")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(new)
        try:
            compile(new, str(p), "exec")
        except SyntaxError as e:
            shutil.copy2(bak, p)
            results.append((name, "FAIL", f"patched file failed to compile, restored backup: {e}"))
            return
        results.append((name, "APPLIED", str(p)))
    else:
        results.append((name, status, str(p)))

def pkg_dir(mod):
    spec = importlib.util.find_spec(mod)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])

# ---------------------------------------------------------------- fix 4
SCATTER_CAST = '''\
        # OLALA fix: the triton kernels require activations, gates and expert
        # weights to share a dtype: the compute dtype follows the ACTIVATIONS.
        # Keying it on self.weight.dtype instead is not phase-stable under
        # FSDP mixed precision: the weight's visible dtype can differ between
        # the original forward (bf16 unsharded views) and gradient-checkpoint
        # recompute (fp32 master params), flipping downstream activation
        # dtypes and tripping check_recomputed_tensors_match. Input dtype is
        # set upstream and identical in both phases. No-op when dtypes agree.
        weight = self.weight
        if weight.dtype != inputs.dtype:
            weight = weight.to(inputs.dtype)
        if gates is not None and gates.dtype != inputs.dtype:
            gates = gates.to(inputs.dtype)

'''

def fix_scattermoe(text):
    if "compute dtype follows the ACTIVATIONS" in text:
        return "ALREADY", None
    # Handles three states: pristine upstream, or either variant of the
    # earlier weight-dtype cast (migrated by replacing everything between the
    # forward signature and the parallel_linear call).
    sig = ("    def forward(self, inputs, k, sorted_expert_idxs, sorted_scattered_idxs,\n"
           "                expert_offsets,\n"
           "                gates=None, grouped_in=False, grouped_out=False):\n")
    call_old = "            inputs, self.weight.permute(0, 2, 1), k,"
    call_new = "            inputs, weight.permute(0, 2, 1), k,"
    if sig not in text:
        return "FAIL: ParallelExperts.forward signature not found", None
    sig_end = text.index(sig) + len(sig)
    call_anchor = "        results = parallel_linear("
    call_idx = text.find(call_anchor, sig_end)
    if call_idx < 0:
        return "FAIL: parallel_linear call not found after signature", None
    text = text[:sig_end] + SCATTER_CAST + text[call_idx:]
    if call_old in text:
        text = text.replace(call_old, call_new, 1)
    elif call_new not in text:
        return "FAIL: parallel_linear argument line not recognized", None
    return "apply", text

SCATTER_BWD_CAST = '''\
             gates, output_expanded) = ctx.saved_tensors
            # OLALA fix: the incoming gradient can arrive in a promoted dtype
            # (e.g. fp32 leaking back from a mixed-precision shared-expert /
            # gating path) while the saved forward tensors are bf16; the
            # kernels and matmuls below need uniform dtypes. No-op when they
            # already agree.
            if grad_out.dtype != x.dtype:
                grad_out = grad_out.to(x.dtype)
'''

def fix_scattermoe_bwd(text):
    if "incoming gradient can arrive in a promoted dtype" in text:
        return "ALREADY", None
    anchor = "             gates, output_expanded) = ctx.saved_tensors\n"
    if anchor not in text:
        return "FAIL: backward saved_tensors unpack not found", None
    return "apply", text.replace(anchor, SCATTER_BWD_CAST, 1)

# ---------------------------------------------------------------- fix 5b
# verl colocate weight-sync ZMQ socket: hardcoded shared /tmp + a Ray job id
# that restarts at 01000000 every fresh cluster means a stale socket left by
# ANOTHER user (sticky /tmp, not removable) permanently blocks new runs with
# "ZMQError: Address already in use". Use the per-user temp dir (TMPDIR)
# on BOTH the sender and receiver sides (they must agree).
def _fix_zmq_tmpdir(text, old_line, new_lines):
    if "rl-colocate-zmq" not in text:
        return "SKIP", None
    if "_tempfile.gettempdir()" in text:
        return "ALREADY", None
    if old_line not in text:
        return "FAIL: zmq handle line not found", None
    return "apply", text.replace(old_line, new_lines, 1)

def fix_zmq_sender(text):
    return _fix_zmq_tmpdir(
        text,
        '        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"',
        "        import tempfile as _tempfile\n"
        "        # OLALA fix: per-user temp dir (TMPDIR); see applier notes.\n"
        '        self.zmq_handle = (\n'
        '            f"ipc://{_tempfile.gettempdir()}/rl-colocate-zmq-{job_id}"\n'
        '            f"-replica-{self.replica_rank}-rank-{local_rank}.sock"\n'
        "        )",
    )

def fix_zmq_receiver(text):
    return _fix_zmq_tmpdir(
        text,
        '        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{trainer_rank}.sock"',
        "        import tempfile as _tempfile\n"
        "        # OLALA fix: per-user temp dir (TMPDIR); must match sender side.\n"
        '        return (\n'
        '            f"ipc://{_tempfile.gettempdir()}/rl-colocate-zmq-{job_id}"\n'
        '            f"-replica-{replica_rank}-rank-{trainer_rank}.sock"\n'
        "        )",
    )

# ---------------------------------------------------------------- fix 6
def fix_transfer_queue(text):
    if "dropping malformed" in text:
        return "ALREADY", None
    old = ("                request_msg = ZMQMessage.deserialize(serialized_msg)\n"
           "                operation = request_msg.request_type")
    if old not in text:
        return "FAIL: deserialize anchor not found", None
    return "apply", text.replace(old,
        "                # OLALA fix: a malformed frame must not kill the worker\n"
        "                # thread (every later put/get would time out forever).\n"
        "                try:\n"
        "                    request_msg = ZMQMessage.deserialize(serialized_msg)\n"
        "                except Exception as e:\n"
        "                    logger.error(\n"
        "                        f\"[{self.storage_unit_id}]: dropping malformed \"\n"
        "                        f\"request frame: {type(e).__name__}: {e}\")\n"
        "                    continue\n"
        "                operation = request_msg.request_type", 1)

def fix_tq_controller(text):
    if "dropping malformed request frame" in text:
        return "ALREADY", None
    old = ("            messages = self.request_handle_socket.recv_multipart(copy=False)\n"
           "            identity = messages.pop(0)\n"
           "            serialized_msg = messages\n"
           "            request_msg = ZMQMessage.deserialize(serialized_msg)")
    if old not in text:
        return "FAIL: controller request loop anchor not found", None
    return "apply", text.replace(old,
        "            messages = self.request_handle_socket.recv_multipart(copy=False)\n"
        "            identity = messages.pop(0)\n"
        "            serialized_msg = messages\n"
        "            # OLALA fix: a malformed frame must not kill the controller\n"
        "            # loop — every client would then block forever.\n"
        "            try:\n"
        "                request_msg = ZMQMessage.deserialize(serialized_msg)\n"
        "            except Exception as e:\n"
        "                logger.error(\n"
        "                    f\"controller: dropping malformed request frame: \"\n"
        "                    f\"{type(e).__name__}: {e}\")\n"
        "                continue", 1)

# ---------------------------------------------------------------- run
tq_dir = pkg_dir("transfer_queue")
scatter = os.environ.get("OLALA_FIX_SCATTER") or None

patch("scattermoe dtype cast",
      scatter and Path(scatter) / "scattermoe/parallel_experts.py", fix_scattermoe)
patch("scattermoe backward grad cast",
      scatter and Path(scatter) / "scattermoe/parallel_experts.py", fix_scattermoe_bwd)
patch("verl transfer_queue hardening",
      tq_dir and tq_dir / "storage/simple_storage.py", fix_transfer_queue)
patch("verl transfer_queue controller hardening",
      tq_dir and tq_dir / "controller.py", fix_tq_controller)
verl_dir = pkg_dir("verl")
patch("verl zmq socket per-user tmpdir (sender)",
      verl_dir and verl_dir / "workers/rollout/vllm_rollout/vllm_rollout.py", fix_zmq_sender)
patch("verl zmq socket per-user tmpdir (receiver)",
      verl_dir and verl_dir / "workers/rollout/vllm_rollout/utils.py", fix_zmq_receiver)

width = max(len(n) for n, _, _ in results)
fail = False
print()
for name, status, detail in results:
    print(f"  {name:<{width}}  {status:<9} {detail}")
    fail |= status.startswith("FAIL")
print()
sys.exit(1 if fail else 0)
PY
