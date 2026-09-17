# Swift Qwen MTP evaluation

Requested: evaluate bartowski/ukisai_Swift-Qwen3.8-27b-GGUF using embedded MTP, without DFlash.
Pinned HF revision: d8175c5357a565bfd1514ecf6f37201cc8b99acd
Initial file: ukisai_Swift-Qwen3.8-27b-IQ4_XS.gguf (user-selected IQ family; 15.48GB).

Current Lucebox Qwen loader explicitly ignores embedded NextN/MTP blocks (server/src/qwen35/gguf_target_loader.cpp). Use isolated recent ROCm llama.cpp with --spec-type draft-mtp; verify CLI and actual MTP acceptance counters before claiming MTP runs. Do not replace production executable or delete existing models.

Once SSH access is available:
1. Check disk, installed llama.cpp versions and GPU occupancy; download pinned model and record hash. Build isolated compatible llama.cpp if needed.
2. Prepare six fixed tasks with independently checked expected outcomes: arithmetic/reasoning, constraints, text extraction, coding edge cases, tool arguments, tool-result continuation. Keep a short initial iteration; count truncated responses as incomplete, not correct. Do not run generated code on host; use sandbox if execution needed.
3. Capture current Qwen baseline with same prompts and reasoning settings, compression disabled for quality isolation.
4. While idle, stop production with an independent timed restoration guard before loading separate model on GPU. Use one stream, bounded context, GPU layers, and identical sampling for Swift MTP off/on. Record process identity, GGUF template, flags, reasoning mode, input/output/reasoning counts, finish reason, TTFT, wall time and speculative acceptance.
5. Assess Swift off/on to isolate MTP speed; assess current Qwen vs Swift separately (quant/backend differences are confounders). Compare thinking enabled at matched effort; a no-thinking short decode test cannot evaluate reduced overthinking. Do not generalize six tasks into benchmark-level quality claims.
6. Restore production and verify health even if test fails. Provide results and leave experimental endpoint/config distinct pending evidence.

Sources:
https://huggingface.co/bartowski/ukisai_Swift-Qwen3.8-27b-GGUF#mtp
https://huggingface.co/ukisai/Swift-Qwen3.8-27b#optional-mtp-decoding

Author says MTP head is inherited from base model; inclusion does not prove it was retuned. Author efficiency results are not local measurements.
