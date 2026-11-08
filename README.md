# VisoMaster MCP

VisoMaster MCP lets Claude Code, Claude Desktop and Codex run face swaps on a Linux GPU
workstation that has [VisoMaster Fusion](https://github.com/VisoMasterFusion/VisoMaster-Fusion)
installed.

The MCP server runs on your own computer (the client). It sends each request to the
workstation over SSH. On the workstation, GPU work runs as durable background jobs, so a job
keeps running if the client disconnects, and any client can check on it later. The tools return
progress, labelled face crops, previews and finished media.

It is for people who already run, or are prepared to install, VisoMaster Fusion on an NVIDIA
workstation and want to drive it from an AI assistant instead of the Fusion GUI.

## Requirements

| Side | Supported |
| --- | --- |
| Workstation | Linux x86-64 with an NVIDIA GPU, Fusion at the pinned revision, FFmpeg, system Python 3.9 or newer, an OpenSSH server |
| Client | macOS (tested) or Linux (expected to work, not verified), Python 3.11 or newer with [uv](https://docs.astral.sh/uv/), an OpenSSH client with key authentication. Windows is not supported |

All recorded GPU evidence comes from one installation with Maxwell-generation GPUs (compute
capability 5.2, 24 GB class). Other GPUs, drivers and CUDA stacks have not been qualified. The
full list of requirements is in [Configuration reference](docs/reference/configuration.md#platform-requirements).

## Setup

Setup takes four stages. The [installation guide](docs/how-to/install.md) has the full steps.

1. Clone this repository on the client and run `uv sync --locked`.
2. On the workstation, install Fusion at the pinned revision, FFmpeg, a runner script and a
   `remote-config.json` file.
3. Run `scripts/deploy.sh` to publish the workstation code.
4. Register the server in your MCP client with `VISOMASTER_HOST` and `VISOMASTER_REMOTE`, then
   call `health_check`.

When `health_check` reports `compatible: true` and `code_matches: true`, follow the tutorial
[Your first face swap](docs/tutorials/first-face-swap.md).

## Documentation

**Tutorial**

- [Your first face swap](docs/tutorials/first-face-swap.md): detect a face, preview a swap and
  render a short clip.

**How-to guides**

- [Install VisoMaster MCP](docs/how-to/install.md)
- [Register an MCP client](docs/how-to/register-a-client.md)
- [Deploy a new version and roll back](docs/how-to/deploy-a-new-version.md)
- [Upgrade an installation that used built-in defaults](docs/how-to/upgrade-a-legacy-installation.md)
- [Swap faces in a video](docs/how-to/swap-faces.md)
- [Check and weight source images](docs/how-to/check-reference-images.md)
- [Compare settings side by side](docs/how-to/compare-settings.md)
- [Change settings within one render](docs/how-to/vary-settings-within-a-render.md)
- [Resume a failed or cancelled render](docs/how-to/resume-a-render.md)
- [Save and reuse settings with recipes](docs/how-to/use-recipes.md)
- [Free workstation storage](docs/how-to/manage-storage.md)
- [Test, qualify and audit changes](docs/how-to/test-and-qualify.md)

**Reference**

- [Tools](docs/reference/tools.md): all 24 MCP tools, their arguments and results
- [Configuration](docs/reference/configuration.md): client environment variables,
  `remote-config.json`, the runner and the deployment layout
- [Jobs and scheduling](docs/reference/jobs.md): job IDs, states, GPU queue and storage
- [Media requirements](docs/reference/media.md): accepted inputs, render rules and output format
- [Job results and artifacts](docs/reference/results.md): detection, swap reports, reference
  analysis, comparisons, checkpoints and recipes
- [Preflight checks](docs/reference/preflight-checks.md): readiness values and check codes
- [Troubleshooting](docs/reference/troubleshooting.md): error messages and fixes

**Explanation**

- [Architecture](docs/explanation/architecture.md): processes, configuration, releases, job
  control and security boundaries
- [GPU scheduling](docs/explanation/gpu-scheduling.md): how the shared queue orders GPU jobs
- [Face matching](docs/explanation/face-matching.md): how the adapter uses Fusion to find,
  verify and match faces
- [Video processing](docs/explanation/video-processing.md): decoding, encoding, normalization,
  swap reports, segments, comparisons and checkpoints
- [Recipes, preflight and qualification](docs/explanation/reproducibility.md)

**Evidence**

- [VERIFICATION.md](VERIFICATION.md): dated test and GPU qualification records, with their
  limits

## License

The code in this repository is available under the [MIT License](LICENSE).

That license covers only this repository. It does not cover, and does not change the terms of,
anything this project uses. VisoMaster Fusion is licensed under the GNU General Public License
v3.0 (see the `LICENSE` file in its repository). FFmpeg, PyTorch, ONNX Runtime, CUDA and the
other components of Fusion's environment have their own licenses. This project does not include
or redistribute Fusion, its models or FFmpeg; you obtain them yourself under their own terms.

`remote/engine.py` imports Fusion's modules into the same Python process on the workstation.
The MIT License is compatible with the GPL, but if you distribute this adapter together with
Fusion as a combined work, review the GPL's requirements for that distribution.

Model weights carry separate terms that Fusion's model list does not record. In particular,
[InsightFace states](https://github.com/deepinsight/insightface#license) that models trained on
its annotated data are available for non-commercial research purposes only; the Inswapper and
ArcFace recognition models used here come from InsightFace. Check the terms of every model you
download before using its output.

## Responsible use

Face swapping can be used to deceive or to harm the people depicted. Use this project only with
media you have the right to use and with the consent of the people whose likeness you process.
Label synthetic media where appropriate, and follow the laws on synthetic media, privacy and
likeness that apply to you. Fusion's README includes a fuller disclaimer that also applies here.
