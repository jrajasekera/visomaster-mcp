# Configuration reference

Configuration comes from two places. The MCP client registration sets environment variables
that say how to reach the workstation. A file on the workstation, `remote-config.json`, says
where everything is there.

## Platform requirements

| Side | Requirement |
| --- | --- |
| Workstation OS | Linux x86-64 |
| Workstation GPU | An NVIDIA GPU and driver, with `nvidia-smi` |
| Workstation tools | System `python3` 3.9 or newer (standard library only), `git`, `tar`, GNU coreutils, FFmpeg and FFprobe on the `PATH` of noninteractive SSH sessions, an OpenSSH server |
| Fusion | A checkout at revision `f2d6f5ebe190b631ee5c8972332083cf7f8a3e72` with its Python environment and a working CUDA provider |
| Client OS | macOS (tested) or Linux (expected, not verified). Windows is not supported |
| Client tools | Python 3.11 or newer with [uv](https://docs.astral.sh/uv/), an OpenSSH client with key authentication |

Required models: the detector, recognition and swapper models (`Yolov8`, `Inswapper128` and its
ArcFace recognizer by default). Optional features need only their models: face restoration,
expression restoration (LivePortrait) and Fusion's sequential tracking.

Recorded GPU evidence covers one hardware class only: Maxwell architecture, compute capability
5.2, 24 GB VRAM class. See [VERIFICATION.md](../../VERIFICATION.md).

## Client environment

Set these in the MCP client registration. The server validates them on the first tool call, so
clients can list the tools before they are set. `scripts/deploy.sh` reads and validates the
same variables.

| Variable | Required | Default | Rules |
| --- | --- | --- | --- |
| `VISOMASTER_HOST` | yes | none | SSH alias or host name: letters, digits and `._@:%[]-`, no spaces, no leading `-` |
| `VISOMASTER_REMOTE` | yes | none | Absolute workstation deployment directory: letters, digits and `_./-`, no `..` components |
| `VISOMASTER_REMOTE_PYTHON` | no | `python3` | Workstation interpreter for control commands: a command name (letters, digits, `._-`) or an absolute path |
| `VISOMASTER_SSH` | no | `ssh` | Client SSH program: a name or path without control characters |
| `VISOMASTER_DOWNLOADS` | no | `~/Downloads/visomaster-mcp` | Local directory for `download_artifact` and `export_recipe`: absolute or starting with `~`. Relative paths are rejected |

### SSH options

Every call runs `<VISOMASTER_SSH> -T -o BatchMode=yes -o ConnectTimeout=10` with connection
sharing:

| Option | Value |
| --- | --- |
| `ControlMaster` | `auto` |
| `ControlPersist` | `120` (seconds after the last call) |
| `ControlPath` | `~/.ssh/visomaster-%C` |

`--` ends option parsing before the host. Remote commands are quoted with `shlex.join`, and
structured requests travel as JSON on standard input. Nothing relies on SSH environment
forwarding or remote shell profiles.

### Client timeouts

| Call | Timeout |
| --- | --- |
| Most tools | 45 s |
| `job_status` | 45 s plus `wait_seconds` |
| `preflight_swap` | 300 s |
| `resume_render`, `save_recipe`, `run_recipe`, `import_recipe`, `get_recipe` with `verify` | 900 s |
| `upload_media`, `download_artifact` | 3600 s |

Uploads and downloads are foreground calls, so very large transfers also depend on your MCP
client's own tool timeout.

## Remote configuration

The file is `<VISOMASTER_REMOTE>/remote-config.json`. Every workstation process reads it:
control commands, detached supervisors and GPU workers. Each process finds it relative to its
own code, two levels above a release directory, so all clients and reconnects see the same
settings. `VM_CONFIG`, an absolute path, overrides the location; it exists for tests and unusual
layouts.

Start from [`config/remote-config.example.json`](../../config/remote-config.example.json).
Deployment never writes this file.

| Key | Required | Default | Rules |
| --- | --- | --- | --- |
| `version` | yes | | Must be `1` |
| `media_root` | yes | | Absolute path of an existing directory. See [Input paths](media.md#input-paths) |
| `fusion_dir` | no | `<media_root>/tools/visomaster-fusion` | Existing Fusion checkout that contains `.git`. Detection, preview, render and `get_settings` read its layout definition files; normalization does not |
| `state_dir` | no | `<media_root>/runtime/visomaster-mcp` | Absolute path. May lie inside `media_root` or on separate storage, but must not be `media_root`, contain it, or lie inside `fusion_dir`. Created with mode 0700 if missing |
| `gpu` | no | `"0"` | GPU index as a string, without leading zeros. No UUIDs or lists |
| `runner` | yes | | A path, or a list of arguments whose first element is an absolute path to an executable. Never passed through a shell |
| `lock_dir` | no | `state_dir` (also when `null`) | Existing writable directory for the GPU lock and queue |

Unknown keys are rejected.

### Effects

- `gpu` sets `CUDA_VISIBLE_DEVICES` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`) for GPU workers, and
  names the lock file `gpu-<index>.lock`.
- `lock_dir` holds `gpu-<index>.lock`, the queue `gpu-<index>.queue.json` and its lock
  `gpu-<index>.queue.lock`.
- Several deployments on one host that share a GPU must use the same `lock_dir` and the same
  `gpu` index, so their jobs take turns on one lock. A deployment with its own `lock_dir` does
  not coordinate with others.

### Errors and changes

- A missing or invalid file does not stop `health_check`: it reports the problem in
  `config_error` and names the file in `config_path`. Every other tool except `preflight_swap`
  fails with the same message.
- `VM_BASE`, `VM_APP`, `VM_STATE` and `VM_GPU` are no longer read. If one is set on the
  workstation to a value that differs from the file, the configuration is reported as invalid.
- Each job records the SHA-256 of the effective configuration. If the file is changed, made
  invalid or removed while a job is queued, the job fails with an error that asks you to
  resubmit, instead of running with different settings.

## Runner

The runner prepares Fusion's environment and starts the GPU worker. The supervisor starts each
GPU job as:

```text
<runner...> <release>/engine.py <job directory>
```

with `CUDA_VISIBLE_DEVICES` and the job's configuration identity in the environment.

The runner must:

- `exec` Fusion's Python interpreter, so the worker remains the supervisor's direct child.
  Otherwise jobs fail with "The configured runner must exec the Fusion Python interpreter".
- not call `setsid` or `setpgid`, and not daemonize. Cancellation signals the worker's process
  group, which every helper process must stay in.
- keep inherited file descriptors (they hold the GPU lock) and the environment, and leave
  `CUDA_VISIBLE_DEVICES` unchanged.
- exit with the worker's status.

[`config/fusion-runner.example.sh`](../../config/fusion-runner.example.sh) adds the CUDA and
cuDNN libraries installed as wheels inside Fusion's environment to `LD_LIBRARY_PATH`. It has
commented-out options for temporary and cache directories and a PyTorch allocator setting.

If Fusion's environment needs nothing extra, the interpreter itself is a valid runner:

```json
"runner": ["/srv/visomaster/fusion/.venv/bin/python"]
```

GPU workers run with offscreen Qt and two CPU threads.

## Deployment layout

`scripts/deploy.sh` creates this structure under `VISOMASTER_REMOTE`:

| Path | Contents |
| --- | --- |
| `remote-config.json` | The remote configuration (you create it) |
| `releases/<release>/` | One directory per deployed release: `common.py`, `control.py`, `engine.py` |
| `remote` | Symlink to the selected release, `releases/<release>` |

The five newest releases are kept, plus any release an active job still uses. A flat `remote`
directory left by an older deployment is moved to `releases/legacy-<timestamp>` on the next
deployment. The client runs control commands as
`<VISOMASTER_REMOTE_PYTHON> <VISOMASTER_REMOTE>/remote/control.py`.

## Client registration examples

| File | Client |
| --- | --- |
| [`config/claude-desktop.json`](../../config/claude-desktop.json) | Claude Desktop `mcpServers` entry |
| [`config/codex.toml`](../../config/codex.toml) | Codex `mcp_servers` entry |

Both contain placeholders and do not work unmodified. See
[Register an MCP client](../how-to/register-a-client.md).
