FROM python:3.12-slim

WORKDIR /app

# Install Lean 4 (elan) & build tools to compile the native formal verifier kernel
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gcc libc6-dev && \
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:v4.36.0 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.elan/bin:${PATH}"

COPY pyproject.toml README.md ./
COPY lean_kernel ./lean_kernel
COPY moyasser ./moyasser
COPY models ./models
COPY datasets/official_mof_eta/official_sources_manifest.json ./datasets/official_mof_eta/official_sources_manifest.json
COPY training ./training

RUN cd /app/lean_kernel && lake build
RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["uvicorn", "moyasser.app:app", "--host", "0.0.0.0", "--port", "8080"]
