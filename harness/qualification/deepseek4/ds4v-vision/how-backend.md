# Backend how pass

PASS. Read-only explorer at07e3284. Inherited session model.

HTTP entry is server/src/server/http_server.cpp route_request:2127. normalize_chat_messages:1031-1044 retains text blocks and silently discards image_url. render_and_tokenize_request:2043-2056 renders only text. Context validation:2060 counts those tokens. ParsedRequest in http_server.h:269 has original messages and tokens but no image data. GenerateRequest in common/model_backend.h:170 is token-only. DS4 uses the serialworker at http_server.cpp:3917, not the sequence engine. process_job:3926 builds GenerateRequest and dispatches generate/restore_and_generate:4078-4081.

DeepSeek4Backend::generate_from_state:2342 selects fresh or restored prefill. do_prefill:1861 chunks, may split at snapshot or speculative capture boundaries, embeds at2041-2044, and dispatches paired graph at2086-2111. deepseek4_step_layer_range in deepseek4_graph.cpp accepts F32embeddings and original token IDs independently. It expands embeddings into HC at7080-7086. This is the embedding injection boundary. CpuEmbedder in qwen35/gguf_target_loader.cpp:83-93 rejects IDs outside vocabulary; DS4 ignores that return at backend.cpp:2044.

Model loader deepseek4_loader.cpp:237-258 drops unknown global tensors. Binding1830-1900 lacks vision, aligner, image delimiters and bias_vl. Ordinary bias binds at1886. DeepSeek4Layer at internal.h:135 needs the image routing bias.

Routing has several paths. GPU build_moe_routing at graph.cpp:3256-3299; old host hybrid4481-4520; sparse paired host5847-5900; standard layer-major6619-6629; generic layer-range8096-8184. Fused paths5424and5638also use hash routing. Sparse paired path5864explicitly rejects token IDs>=vocab. All image sentinel tokens require learned routing in hash layers. Mixture weights remain unbiased.

Layer-major attention mask at graph.cpp:2023-2050 masks future rows and rows beyond ordinary sliding window. Reference model.py:283-305 gives image tokens left/right visibility across the complete image span. Raw KVrows already combine prior,current,compressed at1924-1928. Preserve compressed visibility while changing raw-image masking. Graph compressor-boundary recursion6973-7022and backend snapshot/capture splitting must not split an image span. A chunk1fallback is incorrect for vision.

Token-only cache lookup at http_server.cpp:3159-3179 and3258-3264will collide for different images with equal layouts. Either incorporate image bytes and preprocessing/model identity or disable multimodal reuse initially. prepare_prompt2949may rewrite tokens through PFlash/FlowKV; PPP2149and3094may rearrange. These need a multimodal policy preserving alignment. process_job3951logs message JSON before truncation, so redact dataURLs.

Backend factory435-455selects monolithic DS4 by default; hybrid expert parallelism lives inside this backend and is distinct from layer split.

No runtime vision claims. Source parity and image HTTP lanes remain mandatory.
