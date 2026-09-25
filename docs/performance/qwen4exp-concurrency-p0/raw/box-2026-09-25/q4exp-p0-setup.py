from pathlib import Path
p=Path('/home/duster/lucebox-qwen4exp/server/CMakeLists.txt')
s=p.read_text()
needle='''    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_qwen4exp_forward.cpp")
        add_executable(smoke_qwen4exp_forward test/smoke_qwen4exp_forward.cpp)
        target_include_directories(smoke_qwen4exp_forward PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
        target_link_libraries(smoke_qwen4exp_forward PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
    endif()
'''
block=needle+'''    add_executable(qwen4exp_p0_memory_probe /tmp/qwen4exp_p0_memory_probe.cpp)
    target_include_directories(qwen4exp_p0_memory_probe PRIVATE ${DFLASH27B_SRC_INCLUDE_DIRS})
    target_link_libraries(qwen4exp_p0_memory_probe PRIVATE dflash_common ggml ${DFLASH27B_GGML_BACKEND_TARGET})
'''
assert needle in s
p.with_suffix('.CMakeLists.p0-backup').write_text(s)
p.write_text(s.replace(needle,block,1))
