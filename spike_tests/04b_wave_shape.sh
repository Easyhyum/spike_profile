#!/usr/bin/env bash
# 04b — same wave count, different tile layout: is the higher power of the large shapes during their FIRST wave a data-reuse
#      effect? With exactly 1 wave (usable SMs CTAs) the per-CTA work is identical whatever M x N; what changes is how the 35/36
#      concurrent CTAs share A / B panels through L2 (a "row" of N-tiles streams a different B panel per CTA, the square grid
#      shares both). SHAPES = list of MxN with (M/256)(N/128) = usable SMs (default: square, one row of N-tiles, one column of
#      M-tiles, two rows); K = WAVE_K (default SIZE); WAVE_CYCLES cycles each. Tags t04_K<k>_M<m>_N<n>_w1.0_<cta>cta_<layout>.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${HW_JSON:-}" ]] && hw_refresh
require gpu pmd scope v_ch_ok ${REQUIRE_I:+i_ch_ok} || exit 0
USABLE="${AUTO_USABLE_SMS:-35}"; : "${WAVE_K:=$SIZE}"
sq="$("$PY" "$HERE/wave_targets.py" 1.0 | awk '{print $3"x"$4}')"                      # the most square 1-wave grid
: "${SHAPES:=$sq 256x$((USABLE*128)) $((USABLE*256))x128 512x$((USABLE*64))}"        # square | 1 row of N-tiles | 1 column | 2 rows
log "04b: 1-wave layouts: $SHAPES  (K in {$WAVE_K}, $USABLE CTAs each, $WAVE_CYCLES cycles)"
for k in $WAVE_K; do
  for shp in $SHAPES; do
    m="${shp%x*}"; n="${shp#*x}"; ctas=$(( (m/256)*(n/128) ))
    lay=$([[ "$shp" == "$sq" ]] && echo square || { [[ $m -eq 256 ]] && echo row || { [[ $n -eq 128 ]] && echo column || echo rows2; }; })
    [[ $((n % 128)) -eq 0 && $((m % 256)) -eq 0 ]] || { log "04b: $shp not tile-aligned -> skipped"; continue; }
    log "04b: K=$k $shp ($lay) -> $ctas CTAs"
    run_eval "t04_K${k}_M${m}_N${n}_w1.0_${ctas}cta_${lay}" -b "$WAVE_CYCLES" $(workload_args | sed "s/--dtype [a-z0-9]*/--dtype fp32/; s/--size [0-9]*/--size $k/") --kernel wave --rows "$m" --cols "$n" $(scope_args) $(trig_args)
  done
done
"$PY" "$HERE/summary_runs.py" "$RESULT_DIR" || true
