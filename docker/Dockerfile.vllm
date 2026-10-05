# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0

FROM astral/uv:0.8.22@sha256:9874eb7afe5ca16c363fe80b294fe700e460df29a55532bbfea234a0f12eddb1 AS uv
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

ARG TARGETARCH
ARG VLLM_COMMIT=81d7293c2167e39f3ffddc9a82d633f94e8a1eaa
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/81d7293c2167e39f3ffddc9a82d633f94e8a1eaa/vllm-0.29.1rc1.dev527%2Bg81d7293c2-cp38-abi3-manylinux_2_28_x86_64.whl
ARG IQUEST_PLUGIN_COMMIT=201be15ed7b5590de2ba0102be5d87b383f71dc6
ARG INSTANTTENSOR_VERSION=0.2.0

LABEL org.opencontainers.image.title="vLLM with the IQuest-Q1 plugin" \
      org.opencontainers.image.source="https://github.com/IQuestLab/vllm-iquest-q1" \
      ai.vllm.source.commit="$VLLM_COMMIT" \
      ai.iquest.plugin.commit="$IQUEST_PLUGIN_COMMIT"

ENV VIRTUAL_ENV=/opt/vllm/.venv \
    PATH="/opt/vllm/.venv/bin:$PATH" \
    VLLM_PLUGINS=iquest_q1 \
    INSTANTTENSOR_BACKEND=AIO_BUFFERED \
    INSTANTTENSOR_IO_DEPTH=32 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/vllm

RUN test "$TARGETARCH" = amd64 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-dev python3-venv build-essential \
        libnuma1 libgomp1 \
        libaio1t64 libibverbs1 \
    && rm -rf /var/lib/apt/lists/* \
    && uv venv "$VIRTUAL_ENV" --python /usr/bin/python3

# Triton compiles Python extensions at runtime, even with prebuilt wheels.
RUN "$VIRTUAL_ENV/bin/python" - <<'PY'
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sysconfig
import tempfile

with tempfile.TemporaryDirectory(dir="/opt/vllm") as directory:
    source = Path(directory) / "_header_check.c"
    extension = source.with_name("_header_check" + sysconfig.get_config_var("EXT_SUFFIX"))
    source.write_text('''
#include <Python.h>
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_header_check", NULL, -1, NULL
};
PyMODINIT_FUNC PyInit__header_check(void) { return PyModule_Create(&module); }
''')
    subprocess.run(
        shlex.split(sysconfig.get_config_var("LDSHARED"))
        + ["-fPIC", "-I" + sysconfig.get_path("include"),
           str(source), "-o", str(extension)],
        check=True,
    )
    spec = importlib.util.spec_from_file_location("_header_check", extension)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
print("Python C extension compile and import check passed")
PY

# The wheel comes from the upstream revision documented by the public plugin.
# These indexes are public and require no credentials.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "$VIRTUAL_ENV/bin/python" \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://flashinfer.ai/whl/ \
        --index-strategy unsafe-best-match \
        "$VLLM_WHEEL_URL" \
        "instanttensor==$INSTANTTENSOR_VERSION" \
        "flashinfer-cubin==0.6.18.post1" \
        "nvidia-cutlass-dsl[cu13]==4.7.1"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps \
        --index-url https://pypi.org/simple \
        "https://github.com/IQuestLab/vllm-iquest-q1/archive/$IQUEST_PLUGIN_COMMIT.tar.gz" \
    && uv pip check --python "$VIRTUAL_ENV/bin/python"

# CPU-only build check; this does not override runtime device selection.
RUN VLLM_TARGET_DEVICE=cpu "$VIRTUAL_ENV/bin/python" - <<'PY'
from dataclasses import fields
from importlib.metadata import version

from transformers import AutoConfig
from vllm.config import SpeculativeConfig
from vllm_iquest_q1 import register

register()
assert AutoConfig.for_model("iquest_q1").model_type == "iquest_q1"
assert AutoConfig.for_model("iquest_q1_mtp").model_type == "iquest_q1_mtp"
assert "draft_sample_method" in {field.name for field in fields(SpeculativeConfig)}
print({name: version(name) for name in ("vllm", "vllm-iquest-q1", "instanttensor")})
PY

EXPOSE 8000
ENTRYPOINT ["/opt/vllm/.venv/bin/vllm", "serve"]
CMD ["--help"]
