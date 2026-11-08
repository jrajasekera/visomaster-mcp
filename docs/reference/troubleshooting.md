# Troubleshooting

Find the message you see, then apply the fix. `health_check` shows most installation problems:
check `config_error`, `compatibility_reason`, `checks` and `code_matches`.

## Connection and configuration

| Message | Fix |
| --- | --- |
| `VISOMASTER_HOST and VISOMASTER_REMOTE not set` (or one of them) | Add both variables to the client registration ([Register an MCP client](../how-to/register-a-client.md)) and start a new session |
| `VISOMASTER_HOST must be an SSH alias or host name ...`, `VISOMASTER_REMOTE must be an absolute workstation path ...` | Fix the value; the allowed characters are in [Client environment](configuration.md#client-environment) |
| `Check VISOMASTER_HOST (...) with: ssh ...` | The alias does not resolve, or key authentication failed. Run the printed command until it succeeds without a prompt |
| `Remote configuration not found at ...` | Create `remote-config.json` in the deployment directory ([Install](../how-to/install.md), step 6) |
| `<file>: <key> ...` | Fix the named key in the named file. `health_check` shows `config_path` and `config_error` |
| `VM_... is no longer read` | Remove the variable from the workstation environment and put the value in `remote-config.json` |
| `checks` shows `git`, `nvidia-smi`, `ffmpeg` or `ffprobe` false | Install the tool, or put it on the `PATH` of noninteractive SSH sessions |
| `code_matches: false` | The deployed code differs from your clone. Deploy with `scripts/deploy.sh`, or check out the version that is deployed |
| `code_matches: null` | The clone's `remote/` files are missing, so the code cannot be compared |

## Fusion and the runner

| Message | Fix |
| --- | --- |
| `Fusion revision changed to ...`, `Fusion app has tracked local changes` or `compatible: false` | Read `compatibility_reason`. Check out the pinned revision and discard tracked changes under `app/`, or qualify them ([Qualify a new Fusion revision](../how-to/test-and-qualify.md#qualify-a-new-fusion-revision)) |
| `The configured runner must exec the Fusion Python interpreter` | End the runner with `exec "$FUSION_PYTHON" "$@"` ([Runner](configuration.md#runner)) |
| `Cannot start the configured runner ...` | Fix the runner path or its permissions |

## Jobs

| Message | Fix |
| --- | --- |
| `job_id already exists with a different request` | Use a new `job_id`, or repeat the original arguments exactly |
| `Remote configuration changed after submission ...` | The file changed while the job was queued. Resubmit |
| `Supervisor exited or host restarted` | The job's supervisor died. Resubmit it, or resume a render with `resume_render` |
| `No faces matched the target references` | The preview swapped no face. Check that the target comes from the same media and that the face is visible in that frame |
| A render refused for frame rate, timestamps, dimensions or rotation | Convert the video with `normalize_media` ([Fix media that renders reject](../how-to/swap-faces.md#fix-media-that-renders-reject)) |
| `Job ... is still active; cancel it and wait for a terminal state`, `Only completed, failed or cancelled jobs can be deleted` | Cancel the job and wait until `job_status` shows a terminal state |
| `A queued or running job, a resumable failed or cancelled render or a saved recipe uses this job's artifacts as input` | Wait for the other job, or delete the recipe or render that uses the artifacts |
| `Artifacts become available after successful completion` | The job has not completed, or failed. Check `job_status` |
| `Inline artifacts must be PNG/JPEG images smaller than 10 MB` | Use `download_artifact` for other files |

## Deployment

| Message from `deploy.sh` | Fix |
| --- | --- |
| `Refusing to deploy while jobs are active: ...` | Wait for the listed jobs to finish, then run the script again |
| `Cannot list active jobs (... query): ...` | The installed release could not report its jobs. Fix the named error, or use `--force` if no job can be running |
| `Preflight refused release ...` | The new release rejects `remote-config.json`. The error names the problem; the active release is unchanged |
| `The remote installation changed between job queries; retry` | Run the script again |
