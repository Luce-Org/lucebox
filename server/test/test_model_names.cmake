# Exercise the actual CLI before model loading. HTTP routing tests construct
# ServerConfig directly, so cannot detect rejection of unused CLI model names.
set(primary "${CMAKE_CURRENT_BINARY_DIR}/model-name-test-missing-primary.gguf")
set(secondary "${CMAKE_CURRENT_BINARY_DIR}/model-name-test-missing-secondary.gguf")
if(EXISTS "${primary}" OR EXISTS "${secondary}")
    message(FATAL_ERROR "CLI test requires missing model paths")
endif()

function(check_launch name expected_code expected_text)
    execute_process(COMMAND "${SERVER}" ${ARGN}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
        TIMEOUT 10)
    string(FIND "${error}" "${expected_text}" found)
    if(NOT "${result}" STREQUAL "${expected_code}" OR found EQUAL -1)
        message(FATAL_ERROR "${name}: exit=${result}\n${output}\n${error}")
    endif()
endfunction()

# Disabled balancing must reach loading with default names, including when
# the selected primary is the second block. No GPU/model assets are needed.
check_launch(default_primary 1 "failed to detect architecture from ${primary}"
    "${primary}" --target-device ${BACKEND}:0 --model "${secondary}" --target-device ${BACKEND}:1)
check_launch(selected_primary 1 "failed to detect architecture from ${secondary}"
    "${primary}" --target-device ${BACKEND}:0 --model "${secondary}" --target-device ${BACKEND}:1
    --load-balancing-primary-gpu ${BACKEND}:1)
check_launch(unused_reserved_name 1 "failed to detect architecture from ${primary}"
    "${primary}" --model "${secondary}" --model-name auto)
check_launch(balanced_duplicate 2 "--model-name must be unique"
    "${primary}" --model "${secondary}" --load-balancing)
check_launch(balanced_reserved 2 "--model-name must be unique"
    "${primary}" --model-name qwen --model "${secondary}" --model-name auto --load-balancing)
check_launch(balanced_unique 1 "failed to detect architecture from ${primary}"
    "${primary}" --model-name qwen --model "${secondary}" --model-name ds4 --load-balancing)

# Shared policy belongs only to loaded models. Disabled balancing preserves
# one-model CLI features even when an unused block contains other policies.
check_launch(disabled_selected_policy 1 "failed to detect architecture from ${primary}"
    "${primary}" --no-fast-rollback --kvflash 1024
    --model "${secondary}" --spark)
check_launch(disabled_second_policy 1 "failed to detect architecture from ${secondary}"
    "${primary}" --target-device ${BACKEND}:0 --kvflash 1024
    --model "${secondary}" --target-device ${BACKEND}:1 --no-fast-rollback
    --load-balancing-primary-gpu ${BACKEND}:1)
check_launch(balanced_policy_rejected 2 "changes process-wide policy"
    "${primary}" --model-name qwen --no-fast-rollback
    --model "${secondary}" --model-name ds4 --load-balancing)
# One block may own the expert device and the peer-access opt-in.
check_launch(balanced_expert_owner 1 "failed to detect architecture from ${primary}"
    "${primary}" --model-name qwen
    --model "${secondary}" --model-name ds4 --expert-device ${BACKEND}:1 --peer-access --load-balancing)
check_launch(balanced_expert_owners_rejected 2 "only one model block may set them"
    "${primary}" --model-name qwen --peer-access
    --model "${secondary}" --model-name ds4 --expert-device ${BACKEND}:1 --load-balancing)

# Swap residency plans device sets before loading, so a draft needs a device.
check_launch(swap_draft_needs_device 2 "needs an explicit --draft-device"
    "${primary}" --model-name qwen --target-device ${BACKEND}:0
    --model "${secondary}" --model-name ds4 --target-device ${BACKEND}:0 --draft "${secondary}"
    --model-routing name --swap-residency)

# Name routing loads every block, so it applies the balanced-mode name rules.
check_launch(name_routing_duplicate 2 "--model-name must be unique"
    "${primary}" --model "${secondary}" --model-routing name)
check_launch(name_routing_unique 1 "failed to detect architecture from ${primary}"
    "${primary}" --model-name qwen --model "${secondary}" --model-name ds4
    --model-routing name --unknown-model primary)
check_launch(unknown_model_needs_name_routing 2 "--unknown-model requires --model-routing name"
    "${primary}" --model-name qwen --model "${secondary}" --model-name ds4
    --load-balancing --unknown-model primary)
