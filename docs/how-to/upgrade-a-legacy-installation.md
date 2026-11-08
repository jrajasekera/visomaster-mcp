# Upgrade an installation that used built-in defaults

Earlier versions had built-in defaults for the SSH host, deployment directory, media root and
runner, and read `VM_BASE`, `VM_APP`, `VM_STATE` and `VM_GPU` on the workstation. Current
versions assume nothing: the client needs `VISOMASTER_HOST` and `VISOMASTER_REMOTE`, and the
workstation reads `remote-config.json`.

This guide upgrades such an installation in an order that keeps new client sessions working at
every step. For later upgrades, use [Deploy a new version](deploy-a-new-version.md).

## Steps

1. **Record the current values.** Call `health_check` and note `app`, `state`, `input_root` and
   `gpu`. Back up your client registration files.

2. **Add the client variables.** Add `VISOMASTER_HOST` and `VISOMASTER_REMOTE` with your current
   host and deployment directory to every client registration (see
   [Register an MCP client](register-a-client.md)). The old version reads the same variables,
   so nothing changes yet.

3. **Write the remote configuration.** Create `<VISOMASTER_REMOTE>/remote-config.json` with the
   values from step 1:

   | Key | Value |
   | --- | --- |
   | `version` | `1` |
   | `media_root` | `input_root` |
   | `fusion_dir` | `app` |
   | `state_dir` | `state` |
   | `gpu` | `gpu` |
   | `runner` | your existing launcher |

   Leave out `lock_dir`, so the GPU lock stays where it was. The old release ignores this file.

4. **Deploy.** Update the clone, run `uv sync --locked`, wait until `list_jobs` shows nothing
   queued or running, then run `scripts/deploy.sh`. Its preflight refuses to switch if the new
   release rejects your configuration. A flat `remote` directory from the old deployment is
   moved into `releases/` and kept as an ordinary release.

5. **Remove the old variables.** The new release no longer reads `VM_BASE`, `VM_APP`,
   `VM_STATE` or `VM_GPU`. If one of them is still set with a value that differs from
   `remote-config.json`, it reports an error instead of ignoring the variable. Remove them from
   the workstation environment.

6. **Check the result.** Confirm that `health_check` shows the same paths and GPU as in step 1,
   with `config_error: null` and `code_matches: true`, that `list_jobs` still lists earlier
   jobs, and that a short [smoke run](test-and-qualify.md#run-the-gpu-smoke-test) completes.

## Roll back

1. Point `remote` at the previous release on the workstation:

   ```bash
   cd <VISOMASTER_REMOTE> && ln -sfn releases/<previous> remote.next && mv -T remote.next remote
   ```

2. Check out the previous client version.
3. Restore the registration backups if needed.

The old release ignores `remote-config.json`, so the file can stay.
