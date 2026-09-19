# Swift IQ4_XS with MTP on linuxmacan

## Results

Six identical, short, known-answer tasks: arithmetic, ordering constraints, Python aliasing, interval merging, structured extraction, and a weather tool call. Temperature 0, seed 42, medium reasoning, max output 2048. No truncations. This is one screening run per configuration, not a general coding or reasoning benchmark.

| Model / mode | Decode tokens/s | Total wall seconds | Generated tokens | Strict passes |
|---|---:|---:|---:|---:|
| Original Unsloth UD-IQ4_XS, llama.cpp, no speculation | 30.59 | 67.41 | 1991 | 6/6 |
| Swift IQ4_XS, llama.cpp, no speculation | 31.82 | 56.44 | 1675 | 5/6 |
| Swift IQ4_XS, llama.cpp, MTP | 67.96 | 30.21 | 1769 | 5/6 |

Decode throughput is total completion tokens divided by summed server predicted_ms, including thinking tokens. Wall time includes prompt processing. Swift MTP accepted 1283/1470 draft tokens (87.28%). MTP improved decode throughput 2.14x and total task time 1.87x versus Swift without speculation. Final answers match between Swift runs, though reasoning sequences and token counts differ.

Both Swift modes correctly reasoned through Python aliasing but returned the correct printed list as a JSON string instead of a nested JSON array. Strict scoring retains this failure; substantive answers were correct on all six tasks. The original model obeyed all six output formats. This does not establish broad quality equivalence.

Swift without MTP generated 15.9% fewer total tokens than the original on this small sample. Different quantization layouts remain a confounder. The run does not reproduce the author's BF16 multi-seed benchmark claims or measure a precise thinking-token reduction.

An earlier current-Lucebox baseline is retained in current.json but excluded from the table: it produced almost no reasoning despite medium being requested, so it is not a fair test of thinking reduction. The earlier 206 tokens/s HumanEval result used different prompts and reasoning behavior and must not be directly compared with this table.

## Configuration and provenance

Model: bartowski/ukisai_Swift-Qwen3.8-27b-GGUF, revision d8175c5357a565bfd1514ecf6f37201cc8b99acd, file ukisai_Swift-Qwen3.8-27b-IQ4_XS.gguf.
SHA256: f717f0ef80a3c814bdd8650cc3e404dd0441056afb46770932a9c5051e444d6c.
GGUF metadata confirms qwen35 architecture, 65 total blocks and one nextn_predict_layers block.

Isolated llama.cpp b10896, ROCm core-10.0, gfx1201. Exact runtime commit is saved in work/swift-mtp/runtime-revision.txt. One stream, 8192 context, 512 batch/microbatch, all layers GPU, flash attention on, Q8_0 K/V, embedded Jinja template, no DFlash, no prompt compression. MTP flags: --spec-type draft-mtp --spec-draft-n-max 3. Exact launch arguments, responses, timings and acceptance metrics are saved in work/swift-mtp.

Lucebox's existing Qwen loader ignores embedded MTP layers; this was explicitly authorized as a separate llama.cpp test. It does not establish native Lucebox MTP support. Production service was restored and verified healthy/idle after testing. The downloaded model and isolated runtime are retained; production default/model configuration was not changed.

Sources:
- https://huggingface.co/bartowski/ukisai_Swift-Qwen3.8-27b-GGUF#mtp
- https://huggingface.co/ukisai/Swift-Qwen3.8-27b#optional-mtp-decoding

The author says the MTP head is inherited from the base model. Inclusion is not evidence that it was retuned.

## Follow-up: matched article prompts, Swift DFlash2 versus MTP

User explicitly requested Lucebox+DFlash2 after MTP, with identical reasoning level and prompts. Used the ten canonical PROMPTS from Lucebox server/scripts/bench_he.py, shared by both backends, not the abbreviated bench_he_http.py set.

Settings for both: same downloaded Swift IQ4_XS file, explicit reasoning_effort=none, temperature=0, seed=42, max_tokens=256, context=8192, single stream, GPU weights/KV, Q8_0 K/V. One unmeasured eight-token warmup per backend; fresh processes. No prompt compression. Lucebox uses its deployed persistent-drafter binary and existing qwen38-dflash2-q8_0.gguf, block 16. llama.cpp uses embedded MTP with draft maximum 3. Exact arguments and all request bodies are in the article-swift-* artifacts.

| Swift backend | Decode tokens/s | End-to-end tokens/s | Total HTTP seconds | Output tokens |
|---|---:|---:|---:|---:|
| Lucebox + DFlash2 | **189.69** | **157.54** | **10.086** | 1589 |
| llama.cpp + MTP | 68.94 | 56.88 | 27.937 | 1589 |

All ten final answers are byte-identical between backends. All finish reasons are stop; reasoning content is empty in both. DFlash2 is 2.75x faster in decode and 2.77x faster end-to-end. Mean reported per-request DFlash acceptance rate is 81.38%; aggregate MTP draft-token acceptance is 98.67%. These metrics have different definitions and proposal lengths, so acceptance percentages alone are not a speed comparison.

The existing DFlash2 drafter does accelerate Swift effectively on this sample despite not being Swift-specific. This is a comparison of complete backend+speculation configurations, not an isolated comparison of speculation algorithms. One ten-prompt run; generated code was not executed, and identical output does not prove that code is correct. This no-thinking benchmark does not measure Swift's reduction in reasoning length. Previous original-Qwen 206 tokens/s run used a different quantization and 131072 context, so it is a reference, not a controlled fine-tuning comparison.

Production restoration runs automatically after each private test. No Swift default was installed and no production cache-policy change was made.

## Swift + Lucebox/DFlash2 with xhigh thinking

Ten canonical article prompts at temperature 0, seed 42, explicit reasoning_effort=xhigh, output capacity 32768 and context 65536. Same target and DFlash2 files, persistent drafter, block16, Q8 KV, no compression. Lucebox logs confirm thinking=true, started_in_thinking=true, normalized effort=x-high. Full family tier allows 28672 reasoning tokens plus a 4096 answer reserve. All ten completed normally.

- Decode: **118.38 tokens/s** (thinking and answer tokens combined).
- End-to-end: **114.38 tokens/s**.
- Total: **50.474s**, 5773 completion tokens, including 2710 reasoning tokens.

This differs from the no-thinking run's output cap/context and output lengths, so it is a description of the requested xhigh configuration, not a controlled causal estimate of thinking's throughput cost. Article answers have not been executed as code or scored for correctness.

The six short quality tasks were also run with these settings: **6/6 passed**, 115.17 decode tokens/s, 15.529s, 1678 total tokens including 1557 reasoning tokens. JSON was parsed directly from final content without removing markdown fences or coercing types. The aliasing prompt was clarified to explicitly require an array of integer arrays, not a string. Response: {"result": [[9, 1], [2], [9, 1]]}. This pass cannot be attributed solely to xhigh because the format instruction was clarified. The original ambiguous format case was NOT tested on Swift+DFlash2 with thinking off. A valid JSON tool call with correct arguments also passed.

Root cause of earlier near-zero-thinking Lucebox comparison: max_tokens=2048 was below the configured hard_limit_reply_budget=4096, so effort's available phase1 budget was clamped to zero. The original medium baseline is therefore unsuitable for reasoning-quality comparisons. This test fixed the request capacity, not the server's budget semantics.

Production restored and verified healthy/idle. Exact request/response artifacts and aggregate statistics: work/swift-mtp/*xhigh*.json.
