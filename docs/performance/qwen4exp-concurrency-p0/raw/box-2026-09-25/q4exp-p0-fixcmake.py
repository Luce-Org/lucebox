from pathlib import Path
p=Path('/home/duster/lucebox-qwen4exp/server/CMakeLists.txt')
s=p.read_text()
block='''    add_executable(qwen4exp_p0_memory_probe /tmp/qwen4exp_p0_memory_probe.cpp)
    target_include_directories(qwen4exp_p0_memory_probe PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
    target_link_libraries(qwen4exp_p0_memory_probe PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
'''
assert s.count(block) == 2
p.write_text(s.replace(block+block, block, 1))
