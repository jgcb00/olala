#!/usr/bin/env bash
# Build and push the olala-env image.
#
#   ./docker/build.sh                 build, tag from the commit, push
#   PUSH=0 ./docker/build.sh          build only
#   TAG=wip ./docker/build.sh         override the immutable tag
#   NO_CACHE=1 ./docker/build.sh      ignore the layer cache
#   ALLOW_DIRTY=1 ./docker/build.sh   push from a dirty tree (see below)
#
# Run it from anywhere; the build context is always the repo root, because the
# Dockerfile COPYs setup_olala_env.sh, serve.sh and docker/*.
#
# ---------------------------------------------------------------------------
# TAGGING
#
# The commit is the identity of the image, and here that is unusually literal:
# setup_olala_env.sh pins every source it builds from by SHA -- the vLLM fork,
# olala, scattermoe, renderers -- and install/requirements.txt is a frozen,
# non-re-resolvable snapshot. So the repo commit determines the entire env, and
# `<short sha>` is a genuinely immutable tag rather than an approximation of
# one. That is what makes it safe for a Job to pin to.
#
# Three tags go up:
#
#   :<short sha>   immutable. What a Job should reference. Never re-pushed to a
#                  different image, because the same commit builds the same env.
#   :<branch>      a moving pointer for whoever is iterating on that branch,
#                  with the slashes flattened (feat/foo -> feat-foo) because a
#                  docker tag cannot contain one. None on a detached HEAD.
#   :latest        moves only for a clean build of the default branch.
#
# A DIRTY TREE IS NOT PUSHED by default. `:<short sha>` would then name an image
# that no commit can reproduce, which is precisely the property the tag is
# supposed to carry. ALLOW_DIRTY=1 overrides it, and the tag becomes
# `<sha>-dirty-<timestamp>` -- unique, and obviously not a commit. :latest and
# :<branch> never move for a dirty build.
#
# The build also records the commit in OCI labels, so `docker inspect` answers
# the same question without the tag: org.opencontainers.image.revision. Inside
# the image, /opt/olala-env/PINS has the resolved SHAs of every pinned source.
# ---------------------------------------------------------------------------
set -euo pipefail

REGISTRY=${REGISTRY:-git.corp.linguacustodia.com:5050/gcaillaut/olala-env}
IMAGE_NAME=${IMAGE_NAME:-olala}
IMAGE="$REGISTRY/$IMAGE_NAME"

# Push by default, like ../olala-antidoom/k8s/build.sh and
# ../olala-rlhf/dgx/build.sh -- a build nobody can pull is rarely the point.
# PUSH=0 to build only. A dirty tree is refused regardless; see TAGGING above.
PUSH=${PUSH:-1}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }

git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repo: $ROOT"

SHA=$(git rev-parse --short=7 HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}

# A docker tag is [a-zA-Z0-9_][a-zA-Z0-9._-]{0,127} -- no slashes. So a branch
# called feat/foo cannot be a tag as it stands, and `docker build -t img:feat/foo`
# fails outright. Flatten the separators and keep only legal characters.
# Detached HEAD reports "HEAD", which is a legal tag but a meaningless one, so
# it gets no branch tag at all -- the SHA already says everything.
BRANCH_TAG=""
if [ "$BRANCH" != "HEAD" ]; then
    BRANCH_TAG=$(printf '%s' "$BRANCH" | tr '/' '-' | tr -c 'a-zA-Z0-9._-' '-' | cut -c1-128)
fi

# Untracked files count. A file the Dockerfile would COPY but git does not know
# about makes the build unreproducible just as surely as a modified one.
DIRTY=""
if [ -n "$(git status --porcelain)" ]; then DIRTY=1; fi

# ---- Work out the tags -----------------------------------------------------
MOVING_TAGS=()
if [ -n "${TAG:-}" ]; then
    echo ">> TAG overridden: $TAG (no moving tags)"
elif [ -n "$DIRTY" ]; then
    TAG="$SHA-dirty-$(date -u +%Y%m%d%H%M%S)"
    echo ">> WORKING TREE IS DIRTY"
    git status --short | sed 's/^/     /'
    if [ "$PUSH" = "1" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
        die "refusing to push from a dirty tree. Either

         commit, then:   $0
         build only:     PUSH=0 $0
         push anyway:    ALLOW_DIRTY=1 $0
                         (goes up as $TAG; the moving tags stay put)"
    fi
else
    TAG="$SHA"
    if [ -n "$BRANCH_TAG" ]; then MOVING_TAGS+=("$BRANCH_TAG"); fi
    if [ "$BRANCH" = "$DEFAULT_BRANCH" ]; then MOVING_TAGS+=("latest"); fi
fi

TAG_ARGS=(-t "$IMAGE:$TAG")
for t in ${MOVING_TAGS+"${MOVING_TAGS[@]}"}; do TAG_ARGS+=(-t "$IMAGE:$t"); done

# ---- What is about to happen ----------------------------------------------
cat <<INFO
>> image    $IMAGE
>> tags     $TAG${MOVING_TAGS+ ${MOVING_TAGS[*]}}
>> commit   $(git log --oneline -1)
>> context  $ROOT
>> pins in setup_olala_env.sh:
INFO
grep -E '^(VLLM_FORK_REF|VLLM_FORK_BASE|OLALA_REF|SCATTERMOE_REF|RENDERERS_REF)=' \
    setup_olala_env.sh | sed 's/^/     /'

# ---- Build -----------------------------------------------------------------
# --platform linux/amd64 explicitly: the frozen snapshot pins a flash-attn
# wheel by URL and it is a linux_x86_64 cp312 build, so an arm64 image cannot
# exist. Better to fail on the platform flag than 40 minutes into a build.
BUILD_ARGS=(--platform linux/amd64 -f docker/Dockerfile)
if [ "${NO_CACHE:-0}" = "1" ]; then BUILD_ARGS+=(--no-cache); fi

echo ">> building (cold: an hour+ -- the frozen snapshot is ~20 GB of wheels;"
echo "   longer under rootless docker, where each layer commit is slow)"
docker build \
    "${BUILD_ARGS[@]}" \
    --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
    --label "org.opencontainers.image.source=$(git remote get-url origin 2>/dev/null || echo unknown)" \
    --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --label "org.opencontainers.image.title=olala-env" \
    --label "org.opencontainers.image.description=Olala 7A1B training + inference env: vllm (Olala fork), verl, transformers, TRL" \
    "${TAG_ARGS[@]}" \
    "$ROOT"

echo ">> built $IMAGE:$TAG"
docker image inspect "$IMAGE:$TAG" --format '   size {{.Size}} bytes' 2>/dev/null || true

# ---- Push ------------------------------------------------------------------
if [ "$PUSH" != "1" ]; then
    cat <<SUMMARY

>> NOT pushed (PUSH=0). To push this build later:
     docker push $IMAGE:$TAG
SUMMARY
    exit 0
fi

# A `docker push` against a registry you are not logged into fails with a 401
# after uploading nothing, but the message is easy to misread as a permissions
# problem with the project. Say it up front instead.
if ! grep -q "$(echo "$REGISTRY" | cut -d/ -f1)" "${DOCKER_CONFIG:-$HOME/.docker}/config.json" 2>/dev/null; then
    echo ">> WARN: no credentials for $(echo "$REGISTRY" | cut -d/ -f1) in your docker config."
    echo "         If the push 401s:  docker login $(echo "$REGISTRY" | cut -d/ -f1)"
fi

for t in "$TAG" ${MOVING_TAGS+"${MOVING_TAGS[@]}"}; do
    echo ">> pushing $IMAGE:$t"
    docker push "$IMAGE:$t"
done

cat <<SUMMARY

>> pushed:
     $IMAGE:$TAG$(for t in ${MOVING_TAGS+"${MOVING_TAGS[@]}"}; do printf '\n     %s' "$IMAGE:$t"; done)

   Pin a Job to the immutable tag, not to :latest:
     image: $IMAGE:$TAG

   Check the image on a GPU node before you rely on it:
     docker run --rm --gpus all $IMAGE:$TAG /opt/olala-env/verify.sh
     docker run --rm --gpus all -v /path/to/export:/model \\
         $IMAGE:$TAG /opt/olala-env/verify.sh /model
SUMMARY
