# Install VisoMaster MCP

This guide installs the MCP server on a client machine and the job-control code on a Linux GPU
workstation, then checks that the two can talk to each other.

Before you start, check the [platform requirements](../reference/configuration.md#platform-requirements).
You need administrator access to the workstation and SSH key access to it from the client.

## Placeholders

The commands below use these placeholders. Replace each one with your own value.

| Placeholder | Meaning |
| --- | --- |
| `<owner>` | GitHub owner of the repository you clone |
| `<checkout>` | Absolute path of your clone on the client machine |
| `gpu-host` | SSH alias of the GPU workstation |
| `/srv/visomaster` | Workstation media root: every source media file lives below it |
| `/srv/visomaster/fusion` | Fusion checkout |
| `/srv/visomaster/mcp` | Deployment directory (`VISOMASTER_REMOTE`) |
| `/srv/visomaster/runtime/mcp` | Job state, uploads and logs (`state_dir`) |
| `/srv/visomaster/bin/fusion-runner` | Runner that starts Fusion's Python |

## What you need

- **Required:** a Fusion checkout at the pinned revision with its Python environment; the
  detector, recognition and swapper models (`Yolov8`, `Inswapper128` and its ArcFace recognizer
  by default); FFmpeg and FFprobe; SSH key access from the client.
- **Optional:** face restoration (`GFPGAN-v1.4` was qualified), expression restoration
  (LivePortrait models) and Fusion's sequential tracking (ByteTrack and landmark smoothing, off
  by default). These need only their Fusion models to be present.

## Steps

1. **Clone and install the client.**

   ```bash
   git clone https://github.com/<owner>/visomaster-mcp.git <checkout>
   cd <checkout>
   uv sync --locked
   ```

   The server runs from this clone as an editable install. Keep the clone: `health_check`
   compares the deployed workstation files with its `remote/` directory, and reports
   `code_matches: null` if those files are missing.

2. **Configure SSH.** Add a `Host gpu-host` entry with key authentication to `~/.ssh/config`.
   Then confirm that noninteractive access works without any prompt:

   ```bash
   ssh -T -o BatchMode=yes gpu-host true
   ```

3. **Install Fusion at the pinned revision** on the workstation. The revision is
   `f2d6f5ebe190b631ee5c8972332083cf7f8a3e72` (`REVISION` in `remote/common.py`).

   ```bash
   git clone https://github.com/VisoMasterFusion/VisoMaster-Fusion /srv/visomaster/fusion
   git -C /srv/visomaster/fusion checkout f2d6f5ebe190b631ee5c8972332083cf7f8a3e72
   ```

   Then follow Fusion's README at that commit ("Non-Portable Installation"): create its Python
   3.12 environment, install the requirements file for your GPU generation, and run
   `python download_models.py`. That script downloads the models listed in
   `app/processors/models_data.py`.

   Confirm that Fusion itself can use your GPU before you continue. Do not edit tracked files
   under `app/`: GPU jobs refuse to run on a modified checkout (see
   [The Fusion revision guard](../explanation/architecture.md#the-fusion-revision-guard)).

4. **Install FFmpeg and FFprobe** on the workstation, for example with `sudo apt install ffmpeg`.
   Both must be on the `PATH` of noninteractive SSH sessions and of the runner.

5. **Create the runner.** Copy [`config/fusion-runner.example.sh`](../../config/fusion-runner.example.sh)
   to `/srv/visomaster/bin/fusion-runner`, adapt the paths in it and make it executable:

   ```bash
   chmod +x /srv/visomaster/bin/fusion-runner
   ```

   The runner must follow the rules in [Runner](../reference/configuration.md#runner). If
   Fusion's environment needs no extra setup, you can skip the script and name Fusion's
   interpreter as the runner in the next step.

6. **Create the remote configuration.** Copy
   [`config/remote-config.example.json`](../../config/remote-config.example.json) to
   `/srv/visomaster/mcp/remote-config.json` and adapt it. Every key is described in
   [Remote configuration](../reference/configuration.md#remote-configuration). Deployment never
   writes this file.

7. **Deploy the workstation code** from the clone on the client:

   ```bash
   export VISOMASTER_HOST=gpu-host VISOMASTER_REMOTE=/srv/visomaster/mcp
   scripts/deploy.sh
   ```

   On a first install it prints `No release is installed; deploying the first one`, and on
   success its last line starts with `Deployed <release> to gpu-host:/srv/visomaster/mcp`. If it
   prints `Preflight refused release ...`, the new release rejected `remote-config.json`; fix
   the error it names and run it again. A warning that the Fusion revision is not the supported
   one does not stop deployment, but GPU jobs will refuse to run until you fix the checkout. [Deploy a new version](deploy-a-new-version.md)
   describes what the script does.

8. **Register the server in your MCP client.** Follow
   [Register an MCP client](register-a-client.md) for Claude Code, Codex or Claude Desktop.

9. **Check the installation.** In a new client session, call `health_check` and
   `get_settings`. Expect:

   - `config_error: null`
   - `compatible: true`
   - `code_matches: true`
   - your GPU listed in `gpus`
   - every boolean in `checks` true (`checks.python` is the workstation Python version)
   - `runner.executable: true`

   If something is wrong, look up the message in [Troubleshooting](../reference/troubleshooting.md).

10. **Run a first job.** Follow the tutorial [Your first face swap](../tutorials/first-face-swap.md),
    or run the automated check in [Test, qualify and audit changes](test-and-qualify.md#run-the-gpu-smoke-test).
