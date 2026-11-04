# Copied from the research prototype prebake/claude-code.Dockerfile (fix-git), the Dockerfile the
# probes built and proved. Only its ARG/FROM lines are normalized to ARG BASE.
# Pre-baked task image for the claude-code arm (Harbor 0.25.0, Claude Code 2.1.312).
# Runs exactly what ClaudeCode.install() runs in a stock trial, so that at trial
# time install()'s first probe (`claude --version` == 2.1.312) returns early.
#
#   docker build -f claude-code.Dockerfile --build-arg BASE_IMAGE=alexgshaw/fix-git:20251031 \
#     -t benchprobe/fix-git__claude-code <empty-context-dir>
#
# Deliberately NOT done, because a stock trial does not do it either:
#   - no ENV (DEBIAN_FRONTEND, PATH, ...): it would persist into every exec
#   - no `rm -rf /var/lib/apt/lists/*` / apt-get clean: stock leaves the lists
#   - no SHELL / WORKDIR / USER / ENTRYPOINT / CMD changes: config stays the task's
#   - /installed-agent is not created: setup() creates it in every trial anyway
# All 27 task images run as root with no USER set. For an image with a USER,
# wrap step 1 in `USER root` ... `USER <original>` and keep step 2 under the
# original user (exec_as_agent runs as the default user).
ARG BASE
FROM ${BASE}

# Provenance only: labels are image metadata, invisible inside the container.
LABEL org.benchprobe.harness="claude-code" \
      org.benchprobe.harness-version="2.1.312" \
      org.benchprobe.harbor-version="0.25.0" \
      org.benchprobe.recipe="harbor/agents/installed/claude_code.py install()"

# 1. base.py ensure_system_dependencies(("curl","bash","nodejs","npm","procps")):
#    probe each command as root, install only the missing packages, in order, with
#    `apt-get update && apt-get install -y ...` under DEBIAN_FRONTEND=noninteractive.
#    (dnf/yum/apk branches omitted: every task image here is apt-based.)
RUN ["bash", "-c", "set -o pipefail; pkgs=(); for pair in curl:curl bash:bash node:nodejs npm:npm pgrep:procps; do command -v \"${pair%%:*}\" >/dev/null 2>&1 || pkgs+=(\"${pair##*:}\"); done; echo \"prebake: missing packages: ${pkgs[*]:-none}\"; if [ ${#pkgs[@]} -gt 0 ]; then DEBIAN_FRONTEND=noninteractive bash -c \"set -o pipefail; apt-get update && apt-get install -y ${pkgs[*]}\"; fi"]

# 2. claude_code.py install(), the exec_as_agent command verbatim
#    (Harbor's _exec adds the leading `set -o pipefail; `).
RUN ["bash", "-c", "set -o pipefail; set -euo pipefail; if command -v apk &> /dev/null; then  npm install -g @anthropic-ai/claude-code@2.1.312; else  curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- 2.1.312; fi && echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && export PATH=\"$HOME/.local/bin:$PATH\" && claude --version"]
