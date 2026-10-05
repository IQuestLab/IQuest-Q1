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
