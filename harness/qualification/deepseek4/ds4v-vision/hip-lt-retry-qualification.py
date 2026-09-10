"""Full-tower numerical qualification. Policies must be frozen/reviewed before launch.

This supervises three sequential Radeon-only lanes through the separate live
operator guard. It makes no isolated performance or HTTP acceptance claim.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys

HOME = Path('/home/marcelorm')
ROOT = HOME / 'lucebox-ds4v-vision-hipblaslt'
BUILD = Path('/tmp/ds4v-lt-hip-build')
SOURCE = '3191e7eaee3f5b4d0caa8e8b228c09a9d52b5f3f'
REFERENCE = HOME / 'ds4v-work/source-rocm210-reference/source-hip-reference'
CPU_REFERENCE = HOME / 'lucebox-ds4v-mix-fix/artifacts/vision-reference'
COMPARE = ROOT / 'server/tools/ds4v_vision/compare.py'
PYTHON = HOME / 'lucebox-ds4v-mix-fix/.venv-vision-reference/bin/python'
FIXED = {
    BUILD / 'ds4v_vision_probe': '71515dbb84a48a764ed109ae55bfe94ebc40f1327bac0ddd9c49ab2268e32be3',
    COMPARE: '9171b902efce4b53a073d0632ded2347daf7de4b6985baa8bcd9cae1a96d694a',
    HOME / 'ds4v-work/ds4v-mmproj.gguf': '58eb6b63243df2db21261ced5568b385b04991f38d45d39b781309497abd4b1c',
    REFERENCE / 'manifest.json': '677b5ef033d009a4c44f9fcf7207276e55c4ec8968044ed237f709a108cb3f86',
    CPU_REFERENCE / 'manifest.json': '38d12f30b9a10ed2f0e99bf4d40a07357ab5e8c4880151c58cb90414e4a53f4f',
    REFERENCE.parent / 'source-hip-reference-freeze.json': '8ab35a8a7adc8c66678af7f50b6de610777f0cf03077e108d1d381957cbe8ce0',
}
LIBRARIES = {
    'libggml-base.so.0': (BUILD / 'ggml/src/libggml-base.so.0', '1bdc3462382f5a208272badc459aee4f1c8c46536241f6e40fdbaee6c7ebeef1'),
    'libggml-cpu.so.0': (BUILD / 'ggml/src/libggml-cpu.so.0', '09c3721cb3e4c3689752fed9163c62dc4e20e5e3136a806a72ef7b544b7f89b3'),
    'libggml-hip.so.0': (BUILD / 'ggml/src/ggml-hip/libggml-hip.so.0', '6c7f4bf85b8fc98c6b1109de5d2de4b5e9beb09680d47f7fd01414e70c556955'),
    'libggml.so.0': (BUILD / 'ggml/src/libggml.so.0', '0c7a845a117b3f57b27b10b7a5a8e8821631e1e7be17374d1e758c6ab407f719'),
    'libhipblaslt.so.1': (Path('/opt/rocm-7.2.4/lib/libhipblaslt.so.1'), '4c89a592944beafe388d21cd491c654820baba3d0c12d35fb8fe7020722bb950'),
    'libamdhip64.so.7': (Path('/opt/rocm-7.2.4/lib/libamdhip64.so.7'), 'f1043337461c8e54ee135e95fa979a7d0e4344676ad5b0554652f844f8f098ac'),
}

def require(ok, why):
    if not ok:
        raise RuntimeError(why)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify_pins(pins):
    for path, sha in pins.items():
        require(digest(path) == sha, 'pin changed: ' + str(path))

def guarded_run(command):
    child = subprocess.Popen(command)
    previous = {}
    def interrupted(signum, frame):
        raise InterruptedError(f'qualification interrupted: {signum}')
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        require(child.wait() == 0, 'lane supervision failed')
    finally:
        # The guard handles SIGTERM by stopping/reaping its direct GPU child.
        # Never kill the guard while it might still own a live GPU process.
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        for signum, handler in previous.items():
            signal.signal(signum, handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--parent-radeon-window-released', action='store_true')
    args = parser.parse_args()
    require(sys.flags.isolated and args.parent_radeon_window_released, 'explicit numerical release required')
    require(digest(args.config) == args.config_sha, 'config changed')
    cfg = json.loads(args.config.read_text())
    require(cfg.get('adapter_sha256') == digest(__file__), 'qualification adapter changed')
    linear_path = Path(cfg['linear_receipt']['path'])
    require(digest(linear_path) == cfg['linear_receipt']['sha256'], 'linear acceptance receipt changed')
    linear = json.loads(linear_path.read_text())
    require(linear.get('schema') == 'ds4v-lt-concurrent-linear-proof-v1' and linear.get('pass') is True
            and linear.get('scope') == 'numerical-only-concurrent', 'accepted linear numerical proof required')
    require(linear['source_commit'] == SOURCE and linear['red_commit'] == 'cce69498d6b01541d06fcf77364f68fdbee4627d', 'linear source mismatch')
    require(linear['pins_sha256'] == '832017215ddea57d145e95e31564c1ef7444020a75ba4c9092cefced7b23241b', 'linear pins mismatch')
    require([x['name'] for x in linear['lanes']] == ['redtiny', 'redpatch', 'redqkv', 'greentiny', 'greenpatch', 'greenqkv'], 'six linear lanes required')
    for lane in linear['lanes']:
        red = lane['name'].startswith('red')
        require(lane['guard_exit'] == 0 and lane['child_exit'] == (3 if red else 0)
                and lane['component_numeric_pass'] is True, 'linear lane not accepted')
        require((lane['source_bitwise_mismatches'] > 0) if red else (lane['source_bitwise_mismatches'] == 0), 'wrong linear numerical outcome')
        require(digest(lane['guard_report_path']) == lane['guard_report_sha256'], 'linear guard evidence changed')
    pins = dict(FIXED)
    pins[args.config] = args.config_sha
    pins[linear_path] = cfg['linear_receipt']['sha256']
    pins.update({path: sha for path, sha in LIBRARIES.values()})
    runtime = Path(cfg['comparator_runtime'])
    policy = Path(cfg['qualification_policy'])
    pins.update({runtime: '3a57f18efb7768403c250a316f37b25f3a592a6ef165289e68483c3c357b32cb',
                 policy: '62b3daef05bcaaac175f6907b8748d64952a4bf96baab316bab9b2140451929f'})
    verify_pins(pins)
    pins.update(json.loads(runtime.read_text())['files'])
    for directory in (REFERENCE, CPU_REFERENCE):
        manifest = json.loads((directory / 'manifest.json').read_text())
        require(set(manifest['images']) == {'corn', 'carrots'}, 'fixture set changed')
        for label, entry in manifest['images'].items():
            require(entry['vit_grid'] == ([23, 34] if label == 'corn' else [42, 61]), 'grid changed')
            for stage in ('patches', 'features', 'embeddings'):
                path = directory / entry[stage]['file']
                require(path.parent == directory, 'fixture path escapes reference')
                pins[path] = entry[stage]['sha256']
    require(subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE, 'source changed')
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD', '--'], check=True)
    ldd = subprocess.check_output(['/usr/bin/ldd', str(BUILD / 'ds4v_vision_probe')], text=True)
    require('not found' not in ldd, 'unresolved dependency')
    resolved = dict(re.findall(r'^\s*(\S+) => (/\S+) \(', ldd, re.M))
    for soname, (path, _) in LIBRARIES.items():
        # The standalone probe links the backend libraries directly; the umbrella
        # libggml is built/pinned but omitted by the linker's --as-needed rule.
        if soname == 'libggml.so.0' and soname not in resolved:
            continue
        require(soname in resolved and Path(resolved[soname]).resolve() == path.resolve(), 'wrong effective library: ' + soname)
    verify_pins(pins)
    # Guard code and lane policies are frozen before importing or invoking them.
    guard_dir = Path(cfg['guard_dir'])
    require(guard_dir == HOME / 'ds4v-work/radeon-numerical-retry', 'wrong retry guard directory')
    pins.update({guard_dir / name: sha for name, sha in cfg['guard_pins'].items()})
    verify_pins(pins)
    require(set(cfg['guard_pins']) == {'guard.py', 'host.py', 'receipt.py', 'run.py', 'private_npu_exec.py'}, 'guard pin set incomplete')
    sys.path.insert(0, str(guard_dir))
    from run import load_policy, launch_command
    from host import Host
    from guard import prepare_preflight
    require([x['name'] for x in cfg['lanes']] == ['carrots', 'corn', 'corn-repeat'], 'lane order changed')
    policies = []
    for lane in cfg['lanes']:
        p = load_policy(Path(lane['policy']), lane['sha256'])
        pins[Path(lane['policy'])] = lane['sha256']
        isolation_pins = p.get('isolation_pins', {})
        require('/usr/bin/python3.12' in isolation_pins and
                all(p['component_pins'].get(path) == sha for path, sha in isolation_pins.items()),
                'namespace runtime is not included in guarded component pins')
        for path, sha in p['component_pins'].items():
            require(Path(path) not in pins or pins[Path(path)] == sha, 'lane pin conflicts with qualification pin: ' + path)
            pins[Path(path)] = sha
        label = lane['name'].split('-')[0]
        h, w = (42, 61) if label == 'carrots' else (23, 34)
        out = guard_dir / p['run_name']
        expected = [str(BUILD / 'ds4v_vision_probe'), str(HOME / 'ds4v-work/ds4v-mmproj.gguf'),
                    str(REFERENCE / f'{label}-patches.f32'), str(h), str(w), str(out), label, '0', 'hip:0']
        require(p['command'] == expected and p['expected_exit'] == 0 and p['role'] == 'candidate', 'wrong lane command/role')
        require('hip_fused_bias_launches=67 retained_workspace_bytes=79691776' in p['required_log_lines'], 'dispatch contract missing')
        sizes = {str(out / f'{label}-features.f32'): h*w*1024*4,
                 str(out / f'{label}-embeddings.f32'): ((h+2)//3)*((w+2)//3)*4096*4}
        require(set(p['required_outputs']) == set(sizes), 'wrong output contract')
        require(all(p['required_outputs'][k]['bytes'] == v for k, v in sizes.items()), 'wrong output dimensions')
        policies.append((p, out, label))
    evidence = Path(cfg['evidence'])
    require(evidence.is_absolute() and not evidence.exists(), 'fresh evidence required')
    evidence.mkdir()
    report = {'pass': False, 'scope': 'concurrent numerical-only; CPU portability separate; no HTTP/performance acceptance',
              'source_commit': SOURCE, 'config_sha256': args.config_sha, 'adapter_sha256': digest(__file__),
              'reference': str(REFERENCE), 'policy_sha256': pins[policy], 'linear_receipt': cfg['linear_receipt'], 'lanes': []}
    try:
        for lane, (p, out, label) in zip(cfg['lanes'], policies):
            guarded_run([str(PYTHON), '-I', str(guard_dir / 'run.py'), '--policy', lane['policy'],
                         '--policy-sha', lane['sha256'], '--parent-radeon-window-released'])
            result = json.loads((out / 'guard.json').read_text())
            require(result['pass'] and result['device_proof_verified'], 'guard/result failed')
            require(result.get('namespace_verified') is True, 'private NPU namespace not verified')
            require(result.get('launch_command') == launch_command(p), 'namespace launcher command changed')
            require(result['policy_sha256'] == lane['sha256'] and result['command'] == p['command']
                    and result['exit'] == 0 and result['numeric_verdict'] == 'pending-independent-comparison',
                    'guard report belongs to a different command/policy/outcome')
            require(set(result['output_evidence']) == set(p['required_outputs']), 'guard output set differs')
            for file, meta in result['output_evidence'].items():
                require(meta['bytes'] == p['required_outputs'][file]['bytes'] and Path(file).stat().st_size == meta['bytes'], 'guard output size differs')
                require(digest(file) == meta['sha256'], 'guard output changed before copying')
                pins[Path(file)] = meta['sha256']
            pins[out / 'guard.json'] = digest(out / 'guard.json')
            log = (out / 'child.log').read_text()
            require(re.findall(r'^hip_fused_bias_launches=(\d+) retained_workspace_bytes=(\d+)$', log, re.M) == [('67', '79691776')], 'ambiguous/incomplete full-tower dispatch')
            pins[out / 'child.log'] = digest(out / 'child.log')
            report['lanes'].append({'name': lane['name'], 'policy_path': lane['policy'], 'policy_sha256': lane['sha256'],
                                    'guard_report_path': str(out / 'guard.json'), 'guard_report_sha256': pins[out / 'guard.json'],
                                    'child_log_path': str(out / 'child.log'), 'child_log_sha256': pins[out / 'child.log'],
                                    'command': p['command'], 'launch_command': result['launch_command'],
                                    'namespace_verified': result['namespace_verified'],
                                    'child_exit': result['exit'], 'outputs': result['output_evidence']})
        native, repeat = evidence / 'native', evidence / 'repeat'
        native.mkdir(); repeat.mkdir()
        for index, (_, out, label) in enumerate(policies):
            for stage in ('features', 'embeddings'):
                original = out / f'{label}-{stage}.f32'
                copied = (repeat if index == 2 else native) / f'{label}-{stage}.f32'
                shutil.copyfile(original, copied)
                require(digest(copied) == pins[original], 'copied output differs from guarded output')
                pins[copied] = pins[original]
        for stage in ('features', 'embeddings'):
            original, copied = native / f'carrots-{stage}.f32', repeat / f'carrots-{stage}.f32'
            shutil.copyfile(original, copied)
            require(digest(copied) == pins[original], 'copied repeat carrots differ')
            pins[copied] = pins[original]
        codes = {}
        for name, ref, output in [('target', REFERENCE, native), ('repeat-target', REFERENCE, repeat),
                                  ('native-vs-cpu', CPU_REFERENCE, native), ('source-hip-vs-cpu', CPU_REFERENCE, REFERENCE)]:
            verify_pins(pins)
            with (evidence / f'{name}.log').open('w') as log:
                result = subprocess.run([str(PYTHON), '-I', str(COMPARE), str(ref), str(output), '--output', str(evidence / f'{name}.json')],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=120)
            require(result.returncode in (0, 3), 'comparator execution failed: ' + name)
            codes[name] = result.returncode
        report['comparisons'] = codes
        report['repeat_exact'] = all(digest(native / f'corn-{s}.f32') == digest(repeat / f'corn-{s}.f32') for s in ('features', 'embeddings'))
        verify_pins(pins)
        require(subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE, 'source changed during qualification')
        subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD', '--'], check=True)
        p = policies[-1][0]
        host = Host(p)
        fd = os.open(HOME / 'ds4v-work/radeon-component.lock', os.O_RDWR | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            require(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_nlink == 1, 'untrusted final coordination lock')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _, first, second = prepare_preflight(p, host.snapshot, host.process_identity)
        finally:
            os.close(fd)
        (evidence / 'final-operator.json').write_text(json.dumps({'first': first, 'second': second}, indent=2)+'\n')
        report['pass'] = codes['target'] == codes['repeat-target'] == 0 and report['repeat_exact']
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        (evidence / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['pass'] else 3

if __name__ == '__main__':
    raise SystemExit(main())
