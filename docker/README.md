# Docker images for IQuest-Q1

These Dockerfiles build serving images for vLLM and SGLang. Run the build
commands below from the repository root with Docker BuildKit enabled.
The examples target Linux x86-64 (`linux/amd64`); the vLLM Dockerfile explicitly
requires this architecture. Other architectures are not covered here.

## Build

```bash
docker build --platform linux/amd64 \
  -f docker/Dockerfile.vllm \
  -t vllm-iquest-q1:local .

docker build --platform linux/amd64 \
  -f docker/Dockerfile.sglang \
  --build-arg SGLANG_IMAGE_TAG=iquest-q1-local \
  -t sglang-iquest-q1:local .
```

The default source revisions and selected dependencies are:

| Image | Source revisions | Selected dependencies |
| --- | --- | --- |
| vLLM | vLLM `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`; IQuest-Q1 plugin `201be15ed7b5590de2ba0102be5d87b383f71dc6` | CUDA `13.0.3`; instanttensor `0.2.0`; flashinfer-cubin `0.6.18.post1`; nvidia-cutlass-dsl `4.7.1` |
| SGLang | `Clement-Wang26/sglang` at `4e55dd2dac238274382ed039798c0680b0f053bd` | fastsafetensors `0.4.0`; sglang-kernel `0.4.8` |

Base images are pinned by digest in each Dockerfile. These pins do not form a
complete dependency lockfile. The SGLang image includes the FP8 per-expert scale
loading fix from [PR #42046](https://github.com/sgl-project/sglang/pull/42046),
using a fixed commit from its public fork.

Build arguments can override the defaults. When changing `VLLM_COMMIT`, also
set `VLLM_WHEEL_URL` to the matching wheel; changing the commit argument alone
does not change the installed vLLM wheel.

## Run

Serving requires NVIDIA GPUs, a compatible NVIDIA driver, and NVIDIA Container
Toolkit configured for Docker. The following examples use eight GPUs and
follow the non-MTP deployment settings in the [main README](../README.md#deployment).
Set `MODEL_ROOT` to an absolute path containing the downloaded IQuest-Q1 model
files. If using a Hugging Face cache snapshot with symlinks, make sure the
symlink targets are also accessible inside the container.

```bash
export MODEL_ROOT=/absolute/path/to/IQuest-Q1
```

### vLLM

The image entrypoint is `vllm serve`, so pass the model path and server options
directly after the image name:

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  --mount "type=bind,src=$MODEL_ROOT,dst=/model,readonly" \
  vllm-iquest-q1:local /model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name IQuest-Q1 \
  --tensor-parallel-size 8 \
  --reasoning-parser iquest_q1 \
  --enable-auto-tool-choice \
  --tool-call-parser iquest_q1
```

### SGLang

The image defaults to a Bash shell. Specify the server command explicitly:

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  --mount "type=bind,src=$MODEL_ROOT,dst=/model,readonly" \
  --entrypoint python3 \
  sglang-iquest-q1:local -u -m sglang.launch_server \
  --model-path /model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name IQuest-Q1 \
  --tp-size 8 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --mem-fraction-static 0.85 \
  --disable-prefill-cuda-graph \
  --enable-metrics \
  --tool-call-parser iquest_q1 \
  --reasoning-parser iquest_q1 \
  --enable-torch-compile \
  --load-format fastsafetensors \
  --speculative-use-rejection-sampling
```

For recursive MTP options, see the [main README](../README.md#deployment), using
`/model` as the model path and `/model/mtp` as the draft model path inside the
container.
