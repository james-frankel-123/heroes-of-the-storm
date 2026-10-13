# Proposer sandbox (scilearn, paper 3)

Proposer sessions (`claude -p`) run in a Docker container with no network. They reach
outcome data only through the broker socket. Spec: `paper/hypothesis_system/FRAMING.md`
5.1 and 7.2, `WAVE0_BRIEF.md` items 3 and 10.

## Files

- `Dockerfile`, `entrypoint.sh`, `forwarder.py`: image `scilearn-sandbox:1` (Ubuntu 24.04, python3, git, ripgrep).
  The claude binary is bind-mounted read-only from the host.
- `proxy.py`: host-side HTTP CONNECT proxy on `<session>/proxy.sock`. Allows port 443 to
  `api.anthropic.com`, `claude.ai`, `claude.com`, `*.anthropic.com`, `*.claude.ai`, `*.claude.com`,
  except `mcp-proxy.anthropic.com` (claude.ai connectors). Resolves the name itself and
  refuses loopback, RFC1918, link-local, 100.64/10, IPv6 ULA and other non-global addresses,
  and IP literals. Logs every request to `<session>/logs/proxy.jsonl`.
- `launch_session.py`: creates the session dir, starts the proxy, runs the container.
- `test_escape.py`: escape tests.

## Build the image

```
cd /home/max/heroes-of-the-storm/training/scilearn/sandbox
nice -n 19 docker build -t scilearn-sandbox:1 .
```

## Launch a proposer session

The broker must already be listening on `/home/max/scilearn-sessions/<id>/broker.sock`.
The view builder fills `/home/max/scilearn-sessions/<id>/view` (or pass `--view-src DIR`).

```
python3 /home/max/heroes-of-the-storm/training/scilearn/sandbox/launch_session.py <id> \
    --prompt-file /path/to/prompt.md [--view-src /path/to/view]
```

This runs, inside the sandbox:

```
claude -p --model claude-opus-5-5 --dangerously-skip-permissions \
    --output-format stream-json --verbose --strict-mcp-config --disallowedTools WebFetch,WebSearch
```

with the prompt on stdin. Output goes to `<session>/logs/claude-<stamp>.stream.jsonl` and
`.stderr.log`; transcripts persist in `<session>/claude-projects/`.

Useful flags: `--dry-run` (print the docker command), `--cmd "<bash>"` (run a command instead
of claude), `--no-broker-check` (tests only), `--model`, `--claude-arg X` (repeatable),
`--allow-web` (do not disable WebFetch/WebSearch; not for proposers).

## Container settings

`--network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --read-only
--init --pids-limit 512 --memory 8g --memory-swap 8g --cpus 2`, tmpfs `/tmp` and `/home/hyp`,
commands run at nice 19. Mounts:

| Host | Container | Mode |
|---|---|---|
| `<session>/view` | `/opt/hyp` | ro |
| `<session>/work` | `/work` (cwd) | rw |
| `<session>/broker.sock` | `/run/hyp/broker.sock` | socket |
| `<session>/proxy.sock` | `/run/hyp/proxy.sock` | socket |
| claude binary (resolved from `~/.npm-global/bin/claude`) | `/opt/claude/claude` | ro |
| `/home/max/.claude-account-2` | `/home/hyp/.claude` (`CLAUDE_CONFIG_DIR`) | rw |
| `<session>/claude-projects` | `/home/hyp/.claude/projects` | rw |
| `<session>/claude-state/<name>` | `/home/hyp/.claude/<name>` for every other entry | rw |

Only `.credentials.json` of the account dir is shared. Claude rewrites it by rename, so the
whole directory is mounted (a single-file bind would go stale after a host-side refresh) and
every other entry is masked by a per-session copy. The session gets a trimmed `.claude.json`
(account fields only, no project history) and a settings file that denies WebFetch and
WebSearch. Inside the container a small forwarder listens on `127.0.0.1:3128` and relays to
the proxy socket; `HTTPS_PROXY` points at it.

## Escape tests

```
nice -n 19 python3 /home/max/heroes-of-the-storm/training/scilearn/sandbox/test_escape.py
```

Prints PASS or FAIL per test and exits nonzero on any escape. It plants a canary label file
in `/home/max/scilearn-private` and a marker transcript in the account dir's `projects/`,
and removes both at the end. It uses the session `escape-test` and a dummy echo server as
the broker.
