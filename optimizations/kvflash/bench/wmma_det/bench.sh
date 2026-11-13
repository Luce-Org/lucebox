#!/bin/bash
# WMMA determinism fix bench on lucebox4 (gfx1201, R9700). Three engine arms on one model and flag set, a kernel
# microbench per arm, the HIP red/green of the new kernel test, and the determinism repro (G1 twice on the fixed
# build, 0/55 pick differences = pass). Nothing runs without ~/kvflash-lme/gpu.lock; caches are dropped before every
# timed arm; gfx1201 is proven from the server log by conc_acc_arm.sh (VOID otherwise).
#   usage: bash bench.sh FIX_SHA [TEST_SHA]      (the short shas stage_build_pin.sh printed; TEST_SHA = tests-only commit)
# Arms (conc_acc_arm.sh, kind C under Amendment 3: --paged-attention --max-concurrency 4 --prefix-cache-slots 0,
# --kvflash 16384, --max-ctx 135168, q8_0 K/V, --no-kv-cache --disk-prefix-cache off, temp 0 from the harness):
#   W0     pin rc21_3ae92becac  LUCE_PAGED_WMMA=0       (V_DOT2, known deterministic)
#   W1old  pin rc21_3ae92becac  LUCE_PAGED_WMMA=1       (WMMA before the fix)
#   W1fix  pin wmmadet_<sha>    LUCE_PAGED_WMMA=1       (WMMA, fixed split)
#   W1fix2 pin wmmadet_<sha>    LUCE_PAGED_WMMA=1       (same again: the determinism repro, compare picks with W1fix)
set -uo pipefail
FIX=${1:?FIX_SHA}; TEST=${2:-}
W=$HOME/kvflash-lme; E=$W/tmp/concacc/scripts_night_1104; LOCK=$W/gpu.lock
O=$W/tmp/concacc/wmmadet; mkdir -p "$O"          # bundle dirs must sit under tmp/concacc/<run>/<arm>/bundles
PIN_OLD=$W/pins/rc21_3ae92becac; MD5_OLD=2de94dd53a6082a304933eab09ad1f15
PIN_NEW=$W/pins/wmmadet_$FIX; MD5_NEW=$(awk '/ luce_server$/{print $1}' $PIN_NEW/MD5SUMS)
B_OLD=$W/builds/rc21_src-3ae92becac/build-hip; B_NEW=$W/builds/wmmadet_src-$FIX/build-hip
B_TEST=""; [ -n "$TEST" ] && B_TEST=$W/builds/wmmadet_test_src-$TEST/build-hip
HERE=$(cd "$(dirname "$0")" && pwd)
say() { echo "=== $(date -Is) $*" | tee -a $O/bench.log; }
case "$O" in /tmp|/tmp/*) echo "VOID: /tmp"; exit 2;; esac
for p in $PIN_OLD $PIN_NEW; do (cd $p && md5sum -c --quiet MD5SUMS) || { say "VOID: pin md5 $p"; exit 2; }; done
[ "$(md5sum $PIN_NEW/luce_server | cut -d' ' -f1)" = "$MD5_NEW" ] || { say "VOID: new pin md5"; exit 2; }
for f in $B_OLD/bench_paged_attn_wmma $B_NEW/bench_paged_attn_wmma $B_NEW/test_model_smoke $B_NEW/test_paged_attn_wmma $B_NEW/test_server_unit; do
  [ -x $f ] || { say "VOID: missing $f (stage_build_pin.sh)"; exit 2; }; done
g=$(rocminfo 2>/dev/null | awk '/^  Name: +gfx/{print $2}' | sed -n 1p); [ "$g" = gfx1201 ] || { say "VOID: hip:0 is '$g'"; exit 7; }
say "bench start fix=$FIX test=${TEST:-none} old_md5=$MD5_OLD new_md5=$MD5_NEW"

# ── 1. HIP red/green + unit suites, then the kernel microbench (one lock hold) ──
(
  exec 9>$LOCK; flock -w 7200 9 || { say "ABORT: gpu.lock"; exit 4; }
  pgrep -x luce_server >/dev/null && { say "ABORT: a luce_server runs"; exit 4; }
  export HIP_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm TMPDIR=$W/tmp
  if [ -n "$B_TEST" ]; then
    say "red: new test on the old kernel (tests-only build $TEST), expect FAIL"
    (cd $B_TEST && LUCE_PAGED_WMMA=1 ./test_model_smoke PagedAttention::RowOutputBitsIndependentOfBatch) > $O/red.txt 2>&1; echo "rc=$?" >> $O/red.txt
    say "red rc: $(tail -1 $O/red.txt); DIFF lines: $(grep -c ' DIFF' $O/red.txt)"
  fi
  say "green: new test on the fixed kernel, expect PASS, 0 DIFF lines"
  (cd $B_NEW && LUCE_PAGED_WMMA=1 ./test_model_smoke PagedAttention::RowOutputBitsIndependentOfBatch) > $O/green.txt 2>&1; echo "rc=$?" >> $O/green.txt
  say "green rc: $(tail -1 $O/green.txt); DIFF lines: $(grep -c ' DIFF' $O/green.txt)"
  say "fixed build: test_model_smoke PagedAttention (all), test_paged_attn_wmma both routes + null cases, test_server_unit"
  (cd $B_NEW && ./test_model_smoke PagedAttention) > $O/smoke_paged.txt 2>&1; echo "rc=$?" >> $O/smoke_paged.txt
  (cd $B_NEW && LUCE_PAGED_WMMA=1 ./test_paged_attn_wmma && LUCE_PAGED_WMMA=0 ./test_paged_attn_wmma \
     && LUCE_PAGED_WMMA=1 LUCE_PAGED_NULL_CASES=1 ./test_paged_attn_wmma && LUCE_PAGED_WMMA=0 LUCE_PAGED_NULL_CASES=1 ./test_paged_attn_wmma \
     && python3 -I $W/builds/wmmadet_src-$FIX/server/test/bench/compare_paged_attn.py paged_attn_out_wmma.bin paged_attn_out_vdot2.bin) > $O/wmma_routes.txt 2>&1; echo "rc=$?" >> $O/wmma_routes.txt
  (cd $B_NEW && ./test_server_unit) > $O/server_unit.txt 2>&1; echo "rc=$?" >> $O/server_unit.txt
  say "smoke rc $(tail -1 $O/smoke_paged.txt); wmma routes rc $(tail -1 $O/wmma_routes.txt); server_unit rc $(tail -1 $O/server_unit.txt)"

  # ── 2. kernel microbench (q8_0 K/V; decode-shaped nq=1 and 4, prefill-shaped nq=128) ──
  # old build at its own table (what the old split saw in production); fixed build at the server table (135168
  # tokens = 8448 blocks -> S = 80 blocks) and at its own table (S = 16, the small-config floor).
  for nq in 1 4 128; do
    sudo -n /usr/local/sbin/luce-drop-caches || { say "ABORT: drop-caches"; exit 8; }
    LUCE_PAGED_WMMA=0 $B_OLD/bench_paged_attn_wmma $nq q8_0        > $O/micro_W0_nq$nq.txt 2>&1
    sudo -n /usr/local/sbin/luce-drop-caches
    LUCE_PAGED_WMMA=1 $B_OLD/bench_paged_attn_wmma $nq q8_0        > $O/micro_W1old_nq$nq.txt 2>&1
    sudo -n /usr/local/sbin/luce-drop-caches
    LUCE_PAGED_WMMA=1 $B_NEW/bench_paged_attn_wmma $nq q8_0 135168 > $O/micro_W1fix_nq$nq.txt 2>&1
    sudo -n /usr/local/sbin/luce-drop-caches
    LUCE_PAGED_WMMA=1 $B_NEW/bench_paged_attn_wmma $nq q8_0        > $O/micro_W1fixS16_nq$nq.txt 2>&1
  done
  say "microbench done"; grep -H " ms " $O/micro_*.txt | tee -a $O/bench.log
) || exit $?

# ── 3. engine arms: one fresh server each, G1 gate-screen sessions on two lanes (55 requests). conc_acc_arm.sh takes
#      gpu.lock itself, drops caches, proves gfx1201 and the pin md5s, checks /proc/<pid>/environ. ──
arm() { # NAME PIN MD5 WMMA
  local name=${1:?} pin=${2:?} md5=${3:?} wmma=${4:?}
  say "arm $name pin=$pin LUCE_PAGED_WMMA=$wmma"
  OUT=$O E=$E DATA=$W/coding/data/livecode.json AMEND3=1 PORT=18991 bash $E/conc_acc_arm.sh $name C $pin $md5 \
    "LUCE_KVFLASH_CONCURRENT_TEST=1 LUCE_PAGED_WMMA=$wmma LUCE_KVFLASH_HOST_MIB=28160 LUCE_KVFLASH_SLOT_BUNDLE_DIR=$O/$name/bundles" > $O/$name.run.txt 2>&1
  say "arm $name rc=$?"; tail -3 $O/$name.run.txt | tee -a $O/bench.log
  # the arm's bundle dir (~8 GB) goes away by exact path, as the night chain does
  rm -rf "${O:?}/${name:?}/bundles"
}
arm W0     $PIN_OLD $MD5_OLD 0
arm W1old  $PIN_OLD $MD5_OLD 1
arm W1fix  $PIN_NEW $MD5_NEW 1
arm W1fix2 $PIN_NEW $MD5_NEW 1

# ── 4. readout ──
python3 -I $HERE/report.py $O $E | tee $O/REPORT.txt
touch $O/BENCH_DONE; say "done: $O/REPORT.txt"
