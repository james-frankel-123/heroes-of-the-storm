"""Worker host conventions. Most workers are Windows hosts: ssh lands in cmd and
commands reach WSL through `wsl -e bash -s` / `wsl rsync`. Direct-WSL hosts run sshd
inside WSL itself, so commands go straight to `bash -s` / `rsync`."""
DIRECT_WSL = {"windows-5090-wsl"}


def shell(host):
    """The remote command that reads a bash script on stdin."""
    return "bash -s" if host in DIRECT_WSL else "wsl -e bash -s"


def rsync_path(host):
    return "rsync" if host in DIRECT_WSL else "wsl rsync"
