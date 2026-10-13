# Proposer sandbox for the scientific-learning harness (paper 3).
# The claude binary is bind-mounted read-only from the host at run time;
# this image only carries python3 (for the hyp CLI), git and basic tools.
FROM ubuntu:24.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 ca-certificates git procps less ripgrep \
 && rm -rf /var/lib/apt/lists/*
COPY forwarder.py /usr/local/lib/scilearn/forwarder.py
COPY entrypoint.sh /usr/local/bin/sandbox-entry
RUN chmod 0755 /usr/local/bin/sandbox-entry && chmod 0644 /usr/local/lib/scilearn/forwarder.py
ENTRYPOINT ["/usr/local/bin/sandbox-entry"]
