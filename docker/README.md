# docker/

The env this repo builds, as an image.

```bash
./docker/build.sh                 # build, tag from the commit, push
PUSH=0 ./docker/build.sh          # build only
```

Registry: `git.corp.linguacustodia.com:5050/gcaillaut/olala-env/olala`

| file | what it is |
|---|---|
| `Dockerfile` | the image. Drives `setup_olala_env.sh`; does not re-implement it |
| `build.sh` | build + commit-based tags + push |
| `check_env.py` | the build's smoke test; also runs against a host venv |
| `verify.sh` | check a built image from inside a container |
| `write_pins.sh` | writes `/opt/olala-env/PINS` before the build prunes `.git` |

## What is in the image

`vllm` (the Olala fork, which is the only vLLM that knows `OlalaForCausalLM`),
`verl`, `transformers`, `trl`, plus the mamba3 kernel stack
(tilelang/CuTeDSL/quack), `scattermoe`, `flash-attn`, `ray`, `peft`,
`accelerate`, `datasets` and the `olala` renderer and parsers. Versions and the
resolved source SHAs are in `/opt/olala-env/PINS`.

Not in the image: **the weights**. A checkpoint is a 13 GB raw HF export and is
mounted at runtime.

## Layout inside the image

```
/opt/olala-env/            the repo, laid out as on a dev box
  env/.venv/               the environment. First on PATH
  env/olala/               parsers/, install/, modeling_olala.py
  env/scattermoe/          wired in by a .pth -- do not move it
  env/renderers/           editable install -- do not move it
  serve.sh  setup_olala_env.sh  verify.sh  check_env.py  PINS
/cache/                    JIT + HF caches. Mount an emptyDir here
/workspace/                WORKDIR
```

The venv is first on `PATH`, so `python`, `vllm`, `ray` and `torchrun` are the
env's. `env/scattermoe` and `env/renderers` are reached through a `.pth` file
and an editable install, both recording absolute paths, so **the tree cannot be
relocated** — a `COPY` to a different prefix produces an image that imports the
wrong scattermoe, silently.

## Use it

Serve a checkpoint (`serve.sh` works unchanged — it finds the venv and the
parsers relative to itself):

```bash
docker run --rm --gpus all -p 8010:8010 \
    -v /path/to/export:/model \
    -v /dev/shm:/dev/shm \
    <image> bash -lc 'CKPT=/model /opt/olala-env/serve.sh'
```

Check an image before relying on it:

```bash
docker run --rm --gpus all <image> /opt/olala-env/verify.sh
docker run --rm --gpus all -v /path/to/export:/model \
    <image> /opt/olala-env/verify.sh /model      # + the step-7 checkpoint check
```

Train: the image has `verl` and `trl` but no launcher. Mount your run directory
and drive it as usual — `install/launch_Olala.example.sh` inside
`/opt/olala-env/env/olala/` is the starting point, and the notes at the end of
`setup_olala_env.sh` cover the one-GPU FSDP2 flags.

## Why the base image's torch is not used

`nvcr.io/nvidia/pytorch:26.05-py3` ships torch `2.12.0a0` against CUDA 13.2. The
env pins `torch==2.11.0+cu128`, and the pin is load-bearing three times over:
the vLLM fork installs with `VLLM_USE_PRECOMPILED=1` and those binaries are
built against torch 2.11's C++ ABI; the snapshot's `flash-attn` is a
`torch2.11` wheel pinned by URL; and `install/requirements.txt` is deliberately
not re-resolvable, so it cannot be re-solved onto another torch.

So the image builds the pinned venv **without** system site-packages, and uses
the base image for what surrounds torch: python3.12, the NCCL 2.30 / OpenMPI /
UCX / EFA / RDMA userspace that multi-node verl runs need, nsight, and a devel
CUDA toolchain with a host `g++`.

Three consequences, all of them silent if you get them wrong, and all handled
in the Dockerfile:

* **A CUDA 12.x toolkit is installed alongside 13.2** (nvcc 12.8.93, +791 MB),
  and `CUDA_HOME` points at it. The mamba3 kernels JIT through tilelang/CuTeDSL
  on the *first forward pass*, against a cu128 torch. With only a 13.2 nvcc the
  container starts fine and then dies mid-inference. Both toolkits stay
  installed and `/usr/local/cuda` still resolves to 13.2, so `CUDA_HOME` is set
  explicitly rather than left to the symlink.
* **`LD_LIBRARY_PATH` is rewritten** to drop the base image's
  `dist-packages/torch/lib`. Torch wheels resolve `libtorch`/`libc10` through
  `DT_RUNPATH`, which the loader searches *after* `LD_LIBRARY_PATH` — leaving
  that entry in place makes our torch 2.11 load 2.12's libraries.
* **`PIP_CONSTRAINT` is cleared.** The base image points it at NVIDIA's
  constraint file, which pins torch to the NGC build.

`verify.sh` asserts all three.

## Build shape

One `RUN` per step of `setup_olala_env.sh`, invoked as `ONLY=<n>`, so the image
and the dev box run the same code. Two of the ten steps are skipped:

* **step 7** loads a checkpoint. There is none in the image — `verify.sh` runs
  it against a mounted export instead.
* **step 9** installs `dragon-agentic` editable. Not this repo.

The step *numbers* are therefore load-bearing. If `setup_olala_env.sh` ever
renumbers its steps, `ONLY=<n>` matches nothing and exits 0 — which is what
`check_env.py` is there to catch, and why it is not optional.

A builder stage does the install and a clean stage copies only the result. That
is not tidiness: a file deleted in a later layer still occupies its bytes in the
layer that created it, so the vLLM build tree (several GB), the precompiled base
wheel and the `.git` directories can only be dropped across a stage boundary.
`env/vllm-fork` goes entirely — step 4 installs vLLM non-editable, so once
site-packages has the wheel the checkout is build detritus.

Cold builds are long: most of the time is `uv pip sync` of the ~20 GB snapshot
plus the vLLM install pulling torch 2.11 a second time into a throwaway build
env. Budget an hour of that work, and **considerably more on hippo**, where
docker is rootless — layer commits and even context loads there run tens of
seconds, so measure before you assume something is stuck. `uv`'s cache is a
`RUN --mount=type=cache`, so it is not in the image and warm rebuilds skip the
downloads. `linux/amd64` only: the pinned `flash-attn` wheel is a
`cp312 linux_x86_64` build.

## Tagging

The commit *is* the identity of the image here — `setup_olala_env.sh` pins every
source by SHA and the requirements snapshot is frozen, so the repo commit
determines the whole env.

| tag | |
|---|---|
| `:<short sha>` | immutable. What a Job should pin to |
| `:<branch>` | moving pointer for whoever is iterating |
| `:latest` | moves only for a clean build of `main` |

`build.sh` **refuses to push from a dirty tree** (`ALLOW_DIRTY=1` overrides, and
then the tag says `-dirty-<timestamp>` and the moving tags do not move). The
commit is also recorded as `org.opencontainers.image.revision`, so
`docker inspect` answers the question without the tag.

## Gotchas

* **Mount an `emptyDir` (or tmpfs) at `/cache`.** tilelang and CuTeDSL compile
  the mamba3 kernels on first use. Several replicas compiling into one shared
  RWX directory is a race; baking the cache into the image does not work either,
  since it is device-specific.
* **`HF_HUB_OFFLINE` is not set.** `setup_olala_env.sh` calls it a property of
  the machine rather than of the install, and this image trains as well as
  serves. `serve.sh` defaults it to `1` by itself.
* **`--gpus all` and a big `/dev/shm`** for anything real; the NCCL defaults
  want it.
* **Never run `uv sync` in a container.** Same rule as on the dev box: it would
  reinstall upstream vLLM over the fork. After step 3, only
  `uv pip install --no-deps`.
* `TORCH_CUDA_ARCH_LIST=9.0a` is set **in the builder only**. The mamba3 kernels
  are TileLang/Triton and target the live device, so the image is not restricted
  to Hopper by it — the base image's own value stands at runtime.
