# Register an MCP client

This guide adds the VisoMaster server to Claude Code, Codex or Claude Desktop. Each client
starts its own copy of the server from your clone, and all of them share the same workstation
jobs.

You need a clone with `uv sync --locked` already run, and a deployed workstation (see
[Install VisoMaster MCP](install.md)). The examples use the placeholders listed there.

Every registration must set `VISOMASTER_HOST` and `VISOMASTER_REMOTE`; there are no defaults.
The optional variables are listed in [Client environment](../reference/configuration.md#client-environment).

## Claude Code

1. Add the server:

   ```bash
   claude mcp add --scope user --transport stdio visomaster \
     -e VISOMASTER_HOST=gpu-host -e VISOMASTER_REMOTE=/srv/visomaster/mcp \
     -- <checkout>/.venv/bin/visomaster-mcp
   ```

   Put the server name (`visomaster`) before the `-e` options. `-e` accepts several values and
   would otherwise take the name as one of them. Choose another `--scope` if you want the server
   available only in some projects.

2. Confirm the registration:

   ```bash
   claude mcp get visomaster
   ```

See the [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp) for scopes and
other options.

## Codex

Either run:

```bash
codex mcp add visomaster --env VISOMASTER_HOST=gpu-host \
  --env VISOMASTER_REMOTE=/srv/visomaster/mcp -- <checkout>/.venv/bin/visomaster-mcp
```

or merge the entry in [`config/codex.toml`](../../config/codex.toml) into `~/.codex/config.toml`
and replace its placeholders. That example also sets `startup_timeout_sec = 20` and
`tool_timeout_sec = 120`.

See the [Codex MCP documentation](https://developers.openai.com/codex/mcp/) for details.

## Claude Desktop

1. Open Claude Desktop's configuration file.
2. Merge the `visomaster` entry from [`config/claude-desktop.json`](../../config/claude-desktop.json)
   into `mcpServers`, keeping your other entries.
3. Replace the placeholders. The example files do not work unmodified.
4. Restart Claude Desktop.

## Check the registration

Start a new client session and call `health_check`. Expect `config_error: null`,
`compatible: true` and `code_matches: true`.

The environment variables are checked on the first tool call, not at startup, so the client can
list the tools even when they are missing. If `health_check` reports
`VISOMASTER_HOST and VISOMASTER_REMOTE not set`, or an SSH error, see
[Troubleshooting](../reference/troubleshooting.md).
