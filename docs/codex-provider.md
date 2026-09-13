# Codex Optimizer Provider

Install and authenticate the [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)
before selecting `provider.type: codex`. The transport uses
[`codex exec --json`](https://learn.chatgpt.com/docs/non-interactive-mode) and has
been tested with CLI 0.153.0 and 0.154.0. Authentication is handled by Codex;
credentials do not belong in the YAML configuration.

```yaml
provider:
  type: codex
  capabilities: [read_workspace, edit_workspace, run_commands, load_skills]
  settings:
    model: gpt-5.5
    reasoning_effort: low
    executable: codex
```

`model` and `reasoning_effort` are required; use `null` for the latter to use
Codex's default effort. Choose a model available to your Codex account.
`executable` is optional and defaults to `codex` on `PATH`.

The provider stages `AGENTS.md` and `.agents/skills/`, supplies the session's
instructions explicitly, and disables automatic ancestor instruction loading.
It ignores user CLI configuration to keep provider settings explicit, and runs
with strict configuration so that keys the installed CLI does not recognize are
rejected instead of silently ignored. Codex
writes the existing `.autosaddler/session_output.json` contract, which
AutoSaddler validates against the scenario's schema. Read-only sessions return
the JSON object in their final response instead. This supports the same
schemas as the other providers without requiring OpenAI's narrower structured
output schema format.

Sessions use the workspace-write sandbox when editing is requested and the
read-only sandbox otherwise. Network access and web search are disabled unless
the session requests the `network` capability. Codex uses shell commands for
workspace reads, so these controls are sandbox permissions, not a per-tool
allowlist. No approval bypass flag is used. These sandbox guarantees do not apply
when `sandbox_mode` opts out of the sandbox, as described below.

JSONL events and stderr are exported under `sessions/*/codex-session-state/`,
including on failure. Token usage includes cached-input and reasoning counters
when reported; no dollar cost is inferred. The CLI version is recorded in run
provenance, so upgrading Codex requires a new run ID.

Run the bounded local integration check from the repository root:

```bash
uv run python -m autosaddler.v2.cli \
  --config configs/v2/codex_local_smoke.yaml \
  --run-id codex-local-smoke
```

This uses real Codex sessions and consumes account quota; only the evaluator is
deterministic. Repeating the command resumes the same run. For the Meta-ARE
integration, follow the provisioning steps in the
[GAIA2 smoke run instructions](../README.md#-reproducing-the-included-gaia2-smoke-run)
and replace the smoke config's provider section with the Codex settings above.
Its task agent and judge still require their own API credentials. On Linux,
check the host first as described in the next section.

## Linux hosts that restrict user namespaces

On Linux, Codex enforces its sandbox with bubblewrap, which needs unprivileged
user namespaces. Some hosts restrict them, notably Ubuntu 24.04 and later through
AppArmor. On such hosts every shell command inside a Codex session fails with an
error such as:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
bwrap: setting up uid map: Permission denied
```

Codex then cannot read the workspace, and the session ends without
`.autosaddler/session_output.json`. Once the configured session retries are
exhausted, the run stops with
`Provider session failed after N attempts: Provider completed without the required structured output`.
The session's `codex-session-state/events.jsonl` and final response show the
`bwrap` error.

Check the host before starting a run:

```bash
cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns   # Ubuntu: 1 means restricted
unshare -Ur true && echo "user namespaces available"
```

Choose one of the following. Both require a new run ID for a run that already
exhausted its session retries.

**Allow bubblewrap to create user namespaces.** This keeps the Codex sandbox and
requires root. The narrowest change grants the permission to the system
bubblewrap binary only; Codex prefers `bwrap` on `PATH` over its bundled copy:

```bash
sudo apt install bubblewrap
sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
sudo apparmor_parser -r /etc/apparmor.d/bwrap
```

Alternatively, lift the restriction for the whole host with
`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`. This affects
every user and program on the host.

**Opt out of the Codex sandbox.** Where the host cannot be changed, set:

```yaml
provider:
  type: codex
  settings:
    sandbox_mode: danger-full-access
```

`sandbox_mode` is optional; when set, it must be `danger-full-access`. Codex then
runs model-generated commands with the invoking user's filesystem and network
access, comparable to the Claude provider with `permission_mode: bypassPermissions`.
Use it only on a host or container that is already isolated.
