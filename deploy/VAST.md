# Running the snek3 trainer-server on RunPod or Vast.ai

Use the prebuilt trainer image so the pod starts without cloning the repo or
building Rust/Python dependencies at boot.

## Build and Push

```sh
docker build -t <you>/snek3 -f deploy/Dockerfile .
docker push <you>/snek3
```

CI also builds `ghcr.io/brensch/snek3:latest` from `deploy/Dockerfile`.

## RunPod/Vast Settings

- **Image:** `<you>/snek3` or `ghcr.io/brensch/snek3:latest`
- **GPU:** A100/H100 or similar NVIDIA GPU.
- **Ports:** expose **8050** (RunPod HTTP port / Vast "Open Ports" / `-p 8050:8050`).
- **Storage:** attach persistent storage at `/workspace`. On RunPod Pods, both
  volume disk and network volumes are mounted there. The image writes all run
  artifacts to `/workspace/runs` by default.
- **Command override, optional:** append trainer args such as `--run-id a100-test`
  or `--samples 60000`. If you use a different mount path, include
  `--runs-dir /path/to/runs`.

The image's exec-form entrypoint runs `python -m azsnek.train` directly. Open the
dashboard at the public URL mapped to port 8050 and watch `gen_seconds`,
`samples_per_sec`, and GPU utilization.

## Pulling Run Artifacts

Each run writes everything needed to resume or inspect training under
`/workspace/runs/<run-id>/`, including `state.pt`, replay-buffer shards,
`metrics.jsonl`, dashboard game files, and serving checkpoints.

For quick experiments, RunPod's pod volume disk is enough while the Pod exists.
Download before terminating the Pod, for example:

```sh
rsync -avP root@<pod-host>:/workspace/runs/<run-id>/ ./runs/<run-id>/
```

For long runs, use a RunPod network volume in the same data center as the GPU.
Network volumes persist independently of the Pod and can be attached to another
Pod later, or accessed through RunPod's S3-compatible API for bulk download.
After pausing or stopping a run, archiving the run directory first gives you a
single file to transfer:

```sh
tar -C /workspace/runs -czf /workspace/runs/<run-id>.tgz <run-id>
```

From your local machine, a network volume can also be pulled through the
S3-compatible API:

```sh
aws s3 sync --region <DATACENTER> \
  --endpoint-url https://s3api-<DATACENTER>.runpod.io/ \
  s3://<NETWORK_VOLUME_ID>/runs/<run-id>/ ./runs/<run-id>/
```

## Notes
- The torch cu128 wheel bundles its own CUDA; only the host **driver** must be
  recent enough. GPU runs at native speed under the provider's container runtime.
- Keep the network volume in the same data center as the Pod. RunPod attaches
  network volumes only within their data center.
