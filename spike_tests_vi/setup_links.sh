#!/usr/bin/env bash
# setup_links.sh — (re)create the symlinks of spike_tests_vi to ../spike_tests (run after unpacking the bundle).
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in ../spike_tests/*.sh ../spike_tests/*.py ../spike_tests/launch_hook.c; do
  b="$(basename "$f")"; [[ "$b" == "setup_links.sh" ]] && continue; ln -sfn "$f" "$b"
done
ln -sfn ../spike_tests/legacy legacy; mkdir -p logs results; echo "links ok: $(ls -l | grep -c ' -> ')"
