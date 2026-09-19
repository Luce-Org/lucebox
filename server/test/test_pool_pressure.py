"""Extract the actual legacy allocator; test under a small artificial byte limit.
Pass --hip on linuxmacan to use real hipMalloc/hipFree underneath the limit.
No model, large allocation, or actual device exhaustion is required.
"""
import pathlib,subprocess,tempfile,sys,os
root=pathlib.Path(__file__).resolve().parents[2]
source=(root/'server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
a=source.index('struct ggml_cuda_pool_leg :');b=source.index('// pool with virtual memory',a)
test=root/'server/test'
with tempfile.TemporaryDirectory() as tmp:
 p=pathlib.Path(tmp);cpp=p/'test.cpp';exe=p/'test'
 cpp.write_text((test/'pool_pressure_fixture.h').read_text()+source[a:b]+(test/'pool_pressure_cases.h').read_text())
 flags=['-std=c++17','-DEXPECT_FIXED']
 if '--hip' in sys.argv:flags+=['-DHIP_TEST','-D__HIP_PLATFORM_AMD__','-I/opt/rocm/core-10.0/include','-L/opt/rocm/core-10.0/lib','-Wl,-rpath,/opt/rocm/core-10.0/lib']
 command=[os.environ.get('CXX','c++'),*flags,str(cpp),'-o',str(exe)]
 if '--hip' in sys.argv:command+=['-lamdhip64']
 subprocess.run(command,check=True);subprocess.run([str(exe)],check=True)
