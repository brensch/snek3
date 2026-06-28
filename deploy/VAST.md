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
- **Command override, optional:** append trainer args such as `--run-id a100-test`
  or `--samples 60000`.

The image's exec-form entrypoint runs `python -m azsnek.train` directly. Open the
dashboard at the public URL mapped to port 8050 and watch `gen_seconds`,
`samples_per_sec`, and GPU utilization.

## Notes
- The torch cu128 wheel bundles its own CUDA; only the host **driver** must be
  recent enough. GPU runs at native speed under the provider's container runtime.
- `runs/` lives on the instance. For a speed test you only need the numbers off
  the dashboard; to keep checkpoints, attach a volume or copy `runs/<id>` down
  before terminating the instance.
