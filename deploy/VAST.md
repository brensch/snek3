# Running the snek3 trainer-server on Vast.ai

Two ways. Path A (no image build) is fastest to try; Path B (prebuilt image)
gives quick repeatable starts.

## Path A — base image + on-start script (recommended for a test)

When creating the instance on Vast:

- **Image:** a CUDA base, e.g. `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`
  or any recent `pytorch/pytorch:*-cuda12.*` image. **Pick a host whose driver
  supports CUDA 12.8** (our venv installs torch cu128). If the host driver is
  older, override on-start with `TORCH_INDEX=https://download.pytorch.org/whl/cu124`.
- **Ports:** expose **8050** (Vast → "Open Ports" / `-p 8050:8050`).
- **Env:**
  - `SNEK_SERVE=1` — launch the server after setup
  - `SNEK_SERVE_TOKEN=<your-secret>` — bearer token for write/control/resume
  - `SNEK_RUN_ID=h100-test` — *optional*, auto-start a fresh run (omit to idle)
- **On-start / entrypoint:**

  ```bash
  curl -fsSL https://raw.githubusercontent.com/brensch/snek3/albatross-learning-signal/scripts/h100.sh | bash
  ```

The script installs Rust, builds the extension (`make venv && make build`), then
(because `SNEK_SERVE=1`) runs `make server` in a crash-restart loop in the
foreground, so the instance keeps serving.

Open the dashboard at the public URL Vast maps to port 8050, paste the token,
and watch `gen_seconds` / `samples_per_sec` / GPU%. **Terminate the instance to
stop billing.**

## Path B — prebuilt image

```bash
docker build -t <you>/snek3 -f deploy/Dockerfile .
docker push <you>/snek3
```

On Vast set Image = `<you>/snek3`, expose 8050, env `SNEK_SERVE_TOKEN=…`. The
image's CMD boots the idle server; create runs from the dashboard. Faster cold
starts (no per-boot build), at the cost of building+pushing a multi-GB image.

## Notes
- The torch cu128 wheel bundles its own CUDA; only the host **driver** must be
  recent enough. GPU runs at native speed under Vast's container runtime.
- `runs/` lives on the instance. For a speed test you only need the numbers off
  the dashboard; to keep checkpoints, attach a Vast volume or `scp` `runs/<id>`
  down before terminating.
