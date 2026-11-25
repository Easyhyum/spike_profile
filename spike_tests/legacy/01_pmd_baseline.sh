#!/usr/bin/env bash
# 01 — PMD2X + nvidia-smi + freqmon only (no scope). Baseline power/current/clock of the burst train.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd || exit 0
run_eval t01_pmd -m 0 $(common_args)
