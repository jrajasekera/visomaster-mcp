# Deploy a new version and roll back

This guide updates the workstation code after you pull a new version of this repository, and
shows how to go back to the previous release if something is wrong.

A Git pull updates only the client. The workstation runs its own copy of the three files in
`remote/`, which `scripts/deploy.sh` publishes. [Architecture](../explanation/architecture.md#releases)
explains the release design.

## Deploy

1. Update the clone and its environment:

   ```bash
   cd <checkout>
   git pull
   uv sync --locked
   ```

2. Wait until no job is queued or running. Call `list_jobs` with `status: "queued"` and then
   with `status: "running"`; both should return an empty list. The script refuses to deploy
   while jobs are active, so waiting first saves a failed attempt.

3. Export the connection settings and run the script:

   ```bash
   export VISOMASTER_HOST=gpu-host VISOMASTER_REMOTE=/srv/visomaster/mcp
   scripts/deploy.sh
   ```

   The script reads `VISOMASTER_HOST`, `VISOMASTER_REMOTE`, `VISOMASTER_REMOTE_PYTHON` and
   `VISOMASTER_SSH`, and validates them exactly as the MCP server does.

4. Start a new client session and call `health_check`. Confirm `code_matches: true`,
   `config_error: null` and `compatible: true`.

5. Run the checks that suit the change. For changes to GPU processing, run the
   [GPU smoke test](test-and-qualify.md#run-the-gpu-smoke-test).

### What the script does

1. Asks the installed release for queued and running jobs. It stops if there are any, or if it
   cannot list them. It treats the deployment as a first install only when there is no `remote`
   entry at all.
2. Uploads `common.py`, `control.py` and `engine.py` into a new directory under
   `<VISOMASTER_REMOTE>/releases/`. The directory name combines a hash of the code, a UTC
   timestamp and a random suffix.
3. Runs the new release's health check against the existing `remote-config.json`. If the
   release rejects the configuration, the script removes it, prints
   `Preflight refused release ...` and leaves the active release unchanged.
4. Has the new release promote itself. Under the lock that job submission takes, the release
   checks again for active jobs, points the `remote` symlink at itself and removes all but the
   five newest releases. It never removes the release it selected or one that an active job
   records.

Each job keeps running on the release that accepted it, even after a later deployment.

### Deploy while jobs are active

`scripts/deploy.sh --force` skips both job checks. Use it only when you know no job can be
running, for example when the installed release cannot list its jobs because of an error you
cannot fix first. Release cleanup still keeps the releases that active jobs use, and removes no
release while a job submitted by an older version (one that does not record its release) is
active.

## Roll back

1. On the workstation, list the kept releases, newest last:

   ```bash
   ls -tr /srv/visomaster/mcp/releases
   ```

2. Point `remote` at the previous release. Replace `<previous>` with its directory name:

   ```bash
   cd /srv/visomaster/mcp && ln -sfn releases/<previous> remote.next && mv -T remote.next remote
   ```

3. On the client, check out the previous version of this repository, so that `health_check`
   can compare the code again. Restore backups of your client registrations if you changed them.

4. Call `health_check` and confirm `code_matches: true`.

Releases older than the five newest may already be removed. Leave `remote-config.json` in
place: releases that read it need it, and releases from before it existed ignore it.
