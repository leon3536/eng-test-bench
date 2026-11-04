# Copied from the research prototype prebake/terminus-2/Dockerfile (fix-git), the Dockerfile the
# probes built and proved. Only its ARG/FROM lines are normalized to ARG BASE.
# Pre-baked task image for the terminus-2 arm (Harbor 0.25.0).
#
# Runs, at build time, exactly the one install exec Harbor's
# TmuxSession._install_recording_tools runs at trial time on an apt-get image
# (harbor/agents/terminus_2/tmux_session.py:120-247), including its
# "only what is missing" logic: tmux if `tmux -V` fails, asciinema if
# `asciinema --version` fails (record_terminal_session defaults to True and the
# bench does not override it). On an image that has both it is a no-op, which
# is also what the stock setup does.
#
# Deliberately NOT used, because each would change what `docker inspect` or the
# agent sees compared with a stock container of the task image:
#   ENV (DEBIAN_FRONTEND is inline, as in Harbor), SHELL, USER, WORKDIR, CMD,
#   ENTRYPOINT, LABEL, apt cache cleanup (stock leaves /var/lib/apt/lists).
# The exec form runs `bash -c`, the same wrapper Harbor's docker exec uses
# (environments/docker/docker_unix.py:307-309), without setting SHELL.
#
# Build: docker build --build-arg BASE=alexgshaw/fix-git:20251031 \
#          -t benchprobe/fix-git__terminus-2 .
ARG BASE
FROM ${BASE}
RUN ["bash", "-c", "need=''; tmux -V >/dev/null 2>&1 || need='tmux'; asciinema --version >/dev/null 2>&1 || need=\"${need:+$need }asciinema\"; if [ -n \"$need\" ]; then DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $need; else echo 'tmux and asciinema already present: nothing to bake'; fi"]
