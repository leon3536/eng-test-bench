# Copied from the research prototype prebake/strands/Dockerfile.strands (fix-git), the Dockerfile the
# probes built and proved. Only its ARG/FROM lines are normalized to ARG BASE.
# Pre-baked task image for the bench's strands arm.
#
# Runs, at build time and in Harbor's order, exactly the container operations
# of Harbor 0.25.0 BaseInstalledAgent.setup() + Strands.install() for:
#   strands-agents[all]==1.57.2, factory harnesses/strands_factory
#   (pyproject pins strands-harness==0.3.2), Python 3.12 via uv.
# See stock_install.sh for the per-step source references.
#
# Deliberately NO ENV / SHELL / WORKDIR / USER / LABEL / ENTRYPOINT / CMD:
# the image config (Env, WorkingDir, User, Cmd) stays the task image's.
# Every RUN uses exec form + bash -c, like Harbor's UnixOps exec wrapper, and
# DEBIAN_FRONTEND is exported only inside the apt RUN (Harbor passes it with
# `docker exec -e`), so it does not persist into the container env.
# Nothing is cleaned afterwards (apt lists, uv cache, egg-info): a stock trial
# leaves all of that in the container too.
#
# Build context: the staged dir written by stage_host.py
#   strands_runner.py   (Harbor's runner after Path.write_text on Windows: CRLF)
#   source/             (factory after Harbor's copytree + ignore patterns)
#
#   docker build -f Dockerfile.strands --build-arg TASK_IMAGE=alexgshaw/<task>:20251031 \
#       -t benchprobe/<task>__strands <stage dir>
ARG BASE
FROM ${BASE}

# base.py:1126 setup()
RUN ["bash", "-c", "[ -d /installed-agent ] || mkdir -p /installed-agent"]

# base.py:772 ensure_system_dependencies(("curl", "git")): only the commands that
# are missing, via the first package manager found (all 27 task images are apt).
RUN ["bash", "-c", "set -o pipefail; missing=(); for dep in curl git; do command -v \"$dep\" >/dev/null 2>&1 || missing+=(\"$dep\"); done; if [ ${#missing[@]} -gt 0 ]; then manager=$(for manager in apt-get dnf yum apk; do if command -v \"$manager\" >/dev/null 2>&1; then printf \"%s\" \"$manager\"; break; fi; done); [ \"$manager\" = apt-get ] || { echo \"bake: unsupported package manager '$manager'\" >&2; exit 1; }; export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y \"${missing[@]}\"; fi"]

# strands.py install(): mkdir, upload runner, upload factory dir
RUN ["bash", "-c", "set -o pipefail; mkdir -p /installed-agent/strands"]
COPY --chmod=755 strands_runner.py /installed-agent/strands/strands_runner.py
COPY --chmod=755 source/ /installed-agent/strands/factory/

# docker_unix.py:183 upload_dir() CRLF fix applied when the host is Windows
RUN ["bash", "-c", "find /installed-agent/strands/factory -type f \\( -name '*.sh' -o -name '*.py' -o -name '*.ps1' -o -name '*.cmd' -o -name '*.bat' \\) -exec sed -i 's/\\r$//' {} \\;"]

# strands.py install(): chown -R skipped (environment.default_user is None).
# strands.py install(): exec_as_agent, byte-for-byte Harbor's command string.
RUN ["bash", "-c", "set -o pipefail; set -euo pipefail; curl -LsSf https://astral.sh/uv/install.sh | sh; if [ -f \"$HOME/.local/bin/env\" ]; then . \"$HOME/.local/bin/env\"; fi; uv python install 3.12; uv venv /installed-agent/strands/venv --clear --python 3.12; uv pip install --python /installed-agent/strands/venv/bin/python 'strands-agents[all]==1.57.2' -e /installed-agent/strands/factory"]
