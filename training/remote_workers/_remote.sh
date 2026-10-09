# Shared helpers (sourced). Runs hotsjob.py inside a worker's WSL.
ALL_HOSTS=(max-windows-3090 3080-gaming-desktop windows-5090-wsl)
# Direct-WSL hosts (sshd inside WSL) take `bash -s` / `rsync`; Windows hosts need the wsl prefix.
DIRECT_WSL_HOSTS=" windows-5090-wsl "
remote_shell() { case "$DIRECT_WSL_HOSTS" in *" $1 "*) echo "bash -s" ;; *) echo "wsl -e bash -s" ;; esac; }
remote_rsync() { case "$DIRECT_WSL_HOSTS" in *" $1 "*) echo "rsync" ;; *) echo "wsl rsync" ;; esac; }
hosts_for() { if [ "$1" = all ]; then echo "${ALL_HOSTS[@]}"; else echo "$1"; fi; }
# hotsjob <host> <args...>: args are base64-shipped so no cmd.exe/WSL quoting applies
hotsjob() {
  local host=$1; shift
  local b64; b64=$(printf '%s\0' "$@" | base64 -w0)
  ssh "$host" "$(remote_shell "$host")" <<REMOTE
set -e
source ~/hots/env.sh
echo "$b64" | base64 -d | xargs -0 python ~/hots/repo/training/remote_workers/hotsjob.py
REMOTE
}
