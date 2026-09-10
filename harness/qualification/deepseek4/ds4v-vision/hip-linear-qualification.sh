#!/usr/bin/env bash
# Scoped GPU candidate; execute only after tiny-regression PASS and explicit lane release.
# Standalone component checks may precede text proof in an idle, explicitly released GPU window.
set -euo pipefail
[[ $(hostname) = soulf && $# = 3 && $3 = --gpu-window-released ]] || {
    echo 'Usage on soulf: hip-qualification.sh COMPLETED_TEXT_PROOF_OR_--component-only NEW_EVIDENCE_DIR --gpu-window-released' >&2; exit 2;
}
python="$HOME/lucebox-ds4v-mix-fix/.venv-vision-reference/bin/python"
# No inherited device masks, overrides, GGML/DS4/DFLASH controls, LD_PRELOAD, or Python settings.
exec env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C LC_ALL=C \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$python" -I - "$1" "$2" "$(realpath "$0")" <<'PY'
import hashlib, json, os, re, signal, subprocess, sys, time
from pathlib import Path

home = Path.home()
root = home / 'lucebox-ds4v-linear-rounding'
build = Path('/tmp/ds4v-linear-rounding-scoped-build')
ggml_build = Path('/tmp/ds4v-runtime-hip-build')
binary = build / 'ds4v_vision_probe'
cpu_reference = home / 'lucebox-ds4v-mix-fix/artifacts/vision-reference'
reference = Path('/home/marcelorm/ds4v-work/source-rocm210-reference/source-hip-reference')
freeze_manifest = Path('/home/marcelorm/ds4v-work/source-rocm210-reference/source-hip-reference-freeze.json')
policy_sha = '62b3daef05bcaaac175f6907b8748d64952a4bf96baab316bab9b2140451929f'
mmproj = home / 'ds4v-work/ds4v-mmproj.gguf'
compare = root / 'server/tools/ds4v_vision/compare.py'
component_only = sys.argv[1] == '--component-only'
text_proof = None if component_only else Path(sys.argv[1])
evidence, harness = map(Path, sys.argv[2:])
source_sha = 'be8b0f1b07f1a3a034ce1d7333fd0d3402754c60'
pinned = {
    binary: '79f4928a5172001eef205b100d7578f77560a566d4179fe6755951a15438b483',
    compare: '9171b902efce4b53a073d0632ded2347daf7de4b6985baa8bcd9cae1a96d694a',
    mmproj: '58eb6b63243df2db21261ced5568b385b04991f38d45d39b781309497abd4b1c',
    cpu_reference / 'manifest.json': '38d12f30b9a10ed2f0e99bf4d40a07357ab5e8c4880151c58cb90414e4a53f4f',
    reference / 'manifest.json': '677b5ef033d009a4c44f9fcf7207276e55c4ec8968044ed237f709a108cb3f86',
    freeze_manifest: '8ab35a8a7adc8c66678af7f50b6de610777f0cf03077e108d1d381957cbe8ce0',
    ggml_build / 'ggml/src/libggml-base.so.0': '378b6c81052532d19bd86b32de236553cc1e57f824ce35fb733569470786ed29',
    ggml_build / 'ggml/src/libggml-cpu.so.0': 'b33faf3a600eeff2bea8b692360cff6de397aaf3082ea0c73a9f3a1af9ee70d2',
    ggml_build / 'ggml/src/ggml-hip/libggml-hip.so.0': 'b8991450ee422983b91cfbfcdf8d6b612e92f62f1128c6cce0c6b3e37ff8ef7e',
}

def check(ok, message):
    if not ok: raise RuntimeError(message)

def digest(path):
    with open(path, 'rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def dump(name, value):
    (evidence / name).write_text(json.dumps(value, indent=2) + '\n')

def idle_window():
    bus_env = dict(os.environ, XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}')
    state = subprocess.check_output(['systemctl', '--user', 'show', 'deepseek-dflash.service',
                                     '--property=ActiveState', '--property=MainPID'], env=bus_env, text=True)
    properties = dict(line.split('=', 1) for line in state.splitlines())
    check(properties.get('ActiveState') in ('inactive', 'failed') and properties.get('MainPID') == '0',
          'operator service is not down')
    listeners = subprocess.check_output(['ss', '-ltn'], text=True)
    check(not any(len(row.split()) > 3 and row.split()[3].endswith((':8016', ':8217'))
                  for row in listeners.splitlines()), 'operator or private text port is occupied')
    processes = list(Path('/sys/class/kfd/kfd/proc').iterdir())
    check(not processes, 'GPU compute processes already exist')
    available = int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines()
                         if row.startswith('MemAvailable:'))) * 1024
    check(available >= 8 * 1024**3, 'less than 8 GiB host memory available for component check')
    cards = [p for p in Path('/sys/class/drm').glob('card[0-9]*/device')
             if (p / 'device').is_file() and (p / 'device').read_text().strip() == '0x744c']
    check(len(cards) == 1, 'expected exactly one RX 7900 XT device')
    free_vram = int((cards[0] / 'mem_info_vram_total').read_text()) - int((cards[0] / 'mem_info_vram_used').read_text())
    check(free_vram >= 8 * 1024**3, 'less than 8 GiB discrete VRAM available')
    return {'operator': properties, 'kfd_processes': [], 'host_available_bytes': available,
            'discrete_free_vram_bytes': free_vram}

# No GPU call or evidence mutation before the explicit mode and idle-window gates.
if not component_only:
    text_proof = text_proof.resolve(strict=True)
    allowed = [home / f'{tree}/artifacts/fitter-fix' for tree in
               ('lucebox-ds4v-mix-fix', 'lucebox-ds4v-mix-parallel')]
    check(text_proof.parent in allowed and text_proof.name.startswith('load-proof-'), 'unexpected text-proof directory')
    check((text_proof / 'harness.exit').read_text().strip() == '0', 'private text proof did not complete successfully')
    check(bool((text_proof / 'cleanup.txt').read_text().strip()), 'private text proof cleanup missing')
    verdict = json.loads((text_proof / 'verdict.json').read_text())
    check(all(verdict.get(k) == 'PASS' for k in ('text_load_smoke', 'math_answer', 'longer_decode_speculation')),
          'private text verdict is not PASS')
    server_pid = int((text_proof / 'server.pid').read_text())
    check(server_pid > 1 and not Path(f'/proc/{server_pid}').exists(), 'private text server PID remains present')
initial_window = idle_window()
check(evidence.is_absolute() and not evidence.exists(), 'provide a fresh absolute evidence directory')
evidence.mkdir()  # Parent must already exist; never remove or reuse evidence.
print(f'Evidence: {evidence}', flush=True)
active = None
summary = {'status': 'ISSUES', 'features': 'ISSUES', 'embeddings': 'NOT_QUALIFIED',
           'source_sha': source_sha, 'device': 'hip:0', 'text_proof': str(text_proof) if text_proof else None,
           'scope': '7900XT target-source fidelity only; CPU portability separate; no chat/server acceptance', 'policy_sha256': policy_sha, 'reference': str(reference), 'component_only': component_only,
           'initial_window': initial_window, 'lanes': []}
exit_code = 1

def interrupted(signum, frame):
    raise InterruptedError(signum)

signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)

def memory():
    result = {'monotonic_seconds': time.monotonic(), 'host': Path('/proc/meminfo').read_text()}
    result['drm'] = {str(p): p.read_text().strip() for p in Path('/sys/class/drm').glob('card*/device/mem_info_*')
                     if p.name in ('mem_info_vram_total', 'mem_info_vram_used', 'mem_info_gtt_used')}
    return result

def stop_owned():
    global active
    if active is not None:
        # This unreaped direct child cannot have its PID reused. Never pkill or signal other processes.
        active.terminate()
        try: active.wait(timeout=5)
        except subprocess.TimeoutExpired: active.kill(); active.wait()
        lane = summary['lanes'][-1]
        lane.update(exit=active.returncode, stopped_by_harness=True)
        (evidence / f"{lane['name']}.exit").write_text(str(active.returncode) + '\n')
        dump(f"{lane['name']}.time.json", lane)
        active = None

def run(name, command, timeout=900):
    global active
    lane = {'name': name, 'command': list(map(str, command))}
    summary['lanes'].append(lane)
    started = time.monotonic()
    with open(evidence / f'{name}.log', 'w') as log, open(evidence / f'{name}.memory.jsonl', 'w') as samples:
        active = subprocess.Popen(lane['command'], stdout=log, stderr=subprocess.STDOUT)
        lane['pid'] = active.pid
        (evidence / f'{name}.pid').write_text(str(active.pid) + '\n')
        dump('summary.json', summary)
        while True:
            pid, status, usage = os.wait4(active.pid, os.WNOHANG)
            if pid:
                code = os.waitstatus_to_exitcode(status)
                active.returncode = code
                active = None
                lane.update(exit=code, elapsed_seconds=time.monotonic()-started,
                            user_seconds=usage.ru_utime, system_seconds=usage.ru_stime, max_rss_kib=usage.ru_maxrss)
                (evidence / f'{name}.exit').write_text(str(code) + '\n')
                dump(f'{name}.time.json', lane)
                dump('summary.json', summary)
                return code
            sample = memory()
            try: sample['process_status'] = Path(f'/proc/{active.pid}/status').read_text()
            except FileNotFoundError: pass
            samples.write(json.dumps(sample) + '\n'); samples.flush()
            if time.monotonic()-started > timeout: raise TimeoutError(f'{name} exceeded {timeout}s')
            time.sleep(0.5)

def verify_device(name):
    log = (evidence / f'{name}.log').read_text()
    check(re.search(r'^backend=ROCm0 requested=hip:0$', log, re.M), f'{name}: not the requested HIP backend')
    check(re.search(r'Device 0: [^\n]*7900 XT[^\n]*gfx1100', log), f'{name}: device0 is not 7900 XT/gfx1100')
    check(re.search(r'Device 1: [^\n]*gfx1151', log), f'{name}: device1 order changed')

try:
    check(subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == source_sha,
          'runtime source commit changed')
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--'], check=True)
    for path, sha in pinned.items(): check(digest(path) == sha, f'pinned input changed: {path}')
    manifest = json.loads((reference / 'manifest.json').read_text())
    check(set(manifest['images']) == {'carrots', 'corn'}, 'unexpected fixture set')
    fixtures = {}
    for label, grid in [('carrots', [42, 61]), ('corn', [23, 34])]:
        entry = manifest['images'][label]
        check(entry['vit_grid'] == grid, f'{label}: grid changed')
        for stage in ('patches', 'features', 'embeddings'):
            meta = entry[stage]; path = reference / meta['file']
            check(path.parent == reference and digest(path) == meta['sha256'], f'{label}/{stage}: fixture changed')
            fixtures[str(path)] = meta
    cpu_fixtures = {}
    cpu_manifest = json.loads((cpu_reference / 'manifest.json').read_text())
    check(set(cpu_manifest['images']) == {'carrots', 'corn'}, 'unexpected CPU fixture set')
    for label, entry in cpu_manifest['images'].items():
        for stage in ('patches', 'features', 'embeddings'):
            meta = entry[stage]; path = cpu_reference / meta['file']
            check(path.parent == cpu_reference and digest(path) == meta['sha256'], f'CPU fixture changed: {label}/{stage}')
            cpu_fixtures[str(path)] = meta
    ldd = subprocess.check_output(['/usr/bin/ldd', str(binary)], text=True)
    check('not found' not in ldd and 'libggml-hip.so' in ldd, 'HIP dependency resolution failed')
    (evidence / 'ldd.txt').write_text(ldd)
    libraries = sorted(set(re.findall(r'(?:=>\s+|^\s*)(/[^\s]+)\s+\(', ldd, re.M)))
    software = {path: digest(path) for path in libraries}
    software[str(Path(sys.executable).resolve())] = digest(Path(sys.executable).resolve())
    dump('provenance.json', {'source_sha': source_sha, 'pinned': {str(p): s for p, s in pinned.items()},
                           'shared_libraries_and_python': software, 'fixtures': fixtures, 'cpu_fixtures': cpu_fixtures,
                           'environment': dict(os.environ), 'python': sys.version, 'harness_sha256': digest(harness),
                           'text_proof_files': {p.name: digest(p) for p in text_proof.iterdir() if p.is_file()} if text_proof else {}})
    dump('memory-before.json', memory())
    dump('window-before-gpu.json', idle_window())
    check(run('device-check', [binary, mmproj, '--load-only', '4096', '129280', 'hip:0']) == 0,
          'HIP load-only/device check failed')
    verify_device('device-check')
    native, repeat = evidence / 'native', evidence / 'repeat'
    native.mkdir(); repeat.mkdir()
    for label, h, w, output, lane in [('carrots', 42, 61, native, 'carrots'),
                                    ('corn', 23, 34, native, 'corn'), ('corn', 23, 34, repeat, 'corn-repeat')]:
        check(run(lane, [binary, mmproj, reference / f'{label}-patches.f32', str(h), str(w), output, label, '0', 'hip:0']) == 0,
              f'{lane}: HIP encode failed')
        verify_device(lane)
    # compare.py requires both fixture names: explicitly reuse the first carrots result in repeat comparison.
    for stage in ('features', 'embeddings'):
        (repeat / f'carrots-{stage}.f32').symlink_to(native / f'carrots-{stage}.f32')
    cpu_statuses = [run(name, [sys.executable, '-I', compare, cpu_reference, output, '--output', evidence / f'{name}.json'])
                    for name, output in [('native-vs-cpu', native), ('source-hip-vs-cpu', reference)]]
    check(all(code in (0, 3) for code in cpu_statuses), 'CPU portability comparison execution/shape/hash failure')
    summary['cpu_portability'] = dict(zip(('native_vs_cpu', 'source_hip_vs_cpu'),
                                         ('PASS' if code == 0 else 'ISSUES' for code in cpu_statuses)))
    statuses = [run(name, [sys.executable, '-I', compare, reference, output, '--output', evidence / f'{name}.json'])
                for name, output in [('comparison', native), ('repeat-comparison', repeat)]]
    check(all(code in (0, 3) for code in statuses), 'comparison execution/shape/hash failure')
    comparisons = [json.loads((evidence / f'{name}.json').read_text()) for name in ('comparison', 'repeat-comparison')]
    summary['features'] = 'PASS' if all(c[label]['features']['pass'] for c in comparisons for label in ('carrots', 'corn')) else 'ISSUES'
    summary['embeddings'] = 'PASS' if all(c[label]['embeddings']['pass'] for c in comparisons for label in ('carrots', 'corn')) else 'ISSUES'
    summary['corn_repeat_byte_identical'] = all(digest(native / f'corn-{s}.f32') == digest(repeat / f'corn-{s}.f32')
                                               for s in ('features', 'embeddings'))
    dump('outputs.sha256.json', {str(p.relative_to(evidence)): digest(p) for directory in (native, repeat) for p in directory.glob('*.f32')})
    for path, sha in pinned.items(): check(digest(path) == sha, f'pinned input changed during qualification: {path}')
    for path, meta in fixtures.items(): check(digest(path) == meta['sha256'], f'reference changed during qualification: {path}')
    for path, meta in cpu_fixtures.items(): check(digest(path) == meta['sha256'], f'CPU reference changed during qualification: {path}')
    exit_code = 3 if 3 in statuses or not summary['corn_repeat_byte_identical'] else 0
    summary['status'] = 'PASS' if exit_code == 0 else 'ISSUES'
except InterruptedError as error:
    exit_code = 128 + int(error.args[0]); summary['error'] = f'interrupted by signal {error.args[0]}'
    summary.update(status='ISSUES', features='ISSUES', embeddings='NOT_QUALIFIED')
except TimeoutError as error:
    exit_code = 124; summary['error'] = str(error)
    summary.update(status='ISSUES', features='ISSUES', embeddings='NOT_QUALIFIED')
except Exception as error:
    summary['error'] = f'{type(error).__name__}: {error}'
    summary.update(status='ISSUES', features='ISSUES', embeddings='NOT_QUALIFIED')
finally:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    stop_owned()
    # Once the fixed comparator returned 3, no later command may turn that into success or mask it.
    if any(lane['name'] in ('comparison', 'repeat-comparison') and lane.get('exit') == 3 for lane in summary['lanes']):
        exit_code = 3
    summary['exit'] = exit_code
    dump('summary.json', summary)
    dump('memory-after.json', memory())
    (evidence / 'harness.exit').write_text(str(exit_code) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(exit_code)
PY
