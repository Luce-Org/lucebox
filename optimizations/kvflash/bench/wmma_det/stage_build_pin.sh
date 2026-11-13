#!/bin/bash
# Stage the kvflash/wmma-det tree on lucebox4, build HIP (nice, -j8, no gpu.lock) and make the pin
# ~/kvflash-lme/pins/wmmadet_<sha> (luce_server + ggml libs + MD5SUMS, like pins/rc21_3ae92becac; RUNPATH
# points at the build dir, so builds/wmmadet_src-<sha>/build-hip must stay). Run from the local worktree:
#   bash optimizations/kvflash/bench/wmma_det/stage_build_pin.sh [REF]      (REF defaults to HEAD)
# TEST_ONLY=1 stages the tests-only commit under wmmadet_test_src-<sha> (red arm of the HIP red/green).
# Also builds bench_paged_attn_wmma inside the rc21 build dir (old-kernel microbench); the rc21 pin's md5s
# are checked before and after, the pinned binaries are not rebuilt.
set -euo pipefail
REF=${1:-HEAD}
H=${H:-duster@100.115.193.112}
W=/home/duster/kvflash-lme
SHA=$(git rev-parse --short=9 "$REF")
NAME=wmmadet; [ "${TEST_ONLY:-0}" = 1 ] && NAME=wmmadet_test
S=$W/builds/${NAME}_src-$SHA
echo "staging $REF ($SHA) -> $H:$S"
git archive --format=tar "$REF" | ssh "$H" "mkdir -p $S && tar -x -C $S && echo $SHA > $S/SOURCE_SHA"
ssh "$H" "nice -n 10 bash -s" <<REMOTE
set -euo pipefail
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:\$PATH
export TMPDIR=$W/tmp/${NAME}_$SHA; mkdir -p \$TMPDIR
avail=\$(df --output=avail -BG \$HOME | tail -1 | tr -dc 0-9); [ \$avail -lt 25 ] && { echo "DISK LOW \${avail}G"; exit 3; }
echo "\$(date -Is) $NAME $SHA cfg"
cmake -S $S/server -B $S/build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DLUCE_GPU_BACKEND=hip \
  "-DLUCE_HIP_ARCHITECTURES=gfx1201;gfx1151" "-DCMAKE_HIP_ARCHITECTURES=gfx1201;gfx1151" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DLUCE_FA_ALL_QUANTS=OFF -DLUCE_SERVER=ON -DLUCE_TESTS=ON \
  -DBUILD_TESTING=ON > $S/build-hip.cfg.log 2>&1 || { echo cfg FAILED; tail -20 $S/build-hip.cfg.log; exit 1; }
echo "\$(date -Is) $NAME $SHA build"
cmake --build $S/build-hip --parallel 8 --target luce_server test_server_unit test_model_smoke test_paged_attn_wmma bench_paged_attn_wmma \
  > $S/build-hip.build.log 2>&1 || { echo "build FAILED"; grep -m5 -B2 "error:" $S/build-hip.build.log; exit 2; }
P=$W/pins/${NAME}_$SHA; mkdir -p \$P; cp -p $S/build-hip/luce_server \$P/
for f in \$(ldd $S/build-hip/luce_server | awk '/libggml/{print \$3}'); do cp -pL \$f \$P/; done
(cd \$P && ls | grep -v MD5SUMS | xargs md5sum > MD5SUMS; echo "PIN \$P"; cat MD5SUMS)
echo "\$(date -Is) $NAME $SHA PIN DONE"
if [ "${TEST_ONLY:-0}" != 1 ]; then
  echo "rc21 pin before:"; (cd $W/pins/rc21_3ae92becac && md5sum -c MD5SUMS)
  cmake --build $W/builds/rc21_src-3ae92becac/build-hip --parallel 8 --target bench_paged_attn_wmma > $S/rc21-bench.build.log 2>&1 \
    || { echo "rc21 bench build FAILED"; tail -5 $S/rc21-bench.build.log; exit 2; }
  echo "rc21 pin after:"; (cd $W/pins/rc21_3ae92becac && md5sum -c MD5SUMS)
fi
REMOTE
