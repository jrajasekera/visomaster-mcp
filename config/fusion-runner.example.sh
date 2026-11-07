#!/bin/sh
# Example runner for visomaster-mcp. It does not work unmodified: adapt the values below,
# install it (for example as /srv/visomaster/bin/fusion-runner), chmod +x it and name it in
# remote-config.json "runner".
#
# Invoked as: fusion-runner <release>/engine.py <job-directory>, with CUDA_VISIBLE_DEVICES set.
# Rules: exec the interpreter (the worker must stay the supervisor's direct child), do not
# setsid or daemonize, keep inherited file descriptors and leave CUDA_VISIBLE_DEVICES alone.
set -eu

FUSION_DIR=/srv/visomaster/fusion
FUSION_PYTHON="$FUSION_DIR/.venv/bin/python"

# CUDA and cuDNN libraries installed as pip wheels inside Fusion's environment, if any.
libs=
for lib in "$FUSION_DIR"/.venv/lib/python3.*/site-packages/nvidia/*/lib; do
    if [ -d "$lib" ]; then
        libs="${libs:+$libs:}$lib"
    fi
done
if [ -n "$libs" ]; then
    export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Optional: temporary files and model caches on a large disk.
# export TMPDIR=/srv/visomaster/tmp XDG_CACHE_HOME=/srv/visomaster/cache
# mkdir -p "$TMPDIR"
# Optional: a PyTorch allocator setting that suits your GPU.
# export PYTORCH_CUDA_ALLOC_CONF=backend:native

exec "$FUSION_PYTHON" "$@"
