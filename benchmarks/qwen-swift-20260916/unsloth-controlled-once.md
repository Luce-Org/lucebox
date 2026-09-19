## Regular Unsloth model: one additional run

The regular **Qwen3.8-27B-UD-IQ4_XS** model with DFlash2 finished the ten article prompts in **65.63s**, producing **6,190 tokens**, including **4,895 thinking-text tokens**, at **97.27 tokens/s**. It passed **6/6** fixed quality checks.

This is **one run**, compared with the three-run Swift medians above. The same 16 request bodies, scoring rules, xhigh settings, frozen Swift template, drafter and server binary were used. Input token counts match for every task. Prefix caching and compression remain disabled. The Unsloth model's native template differs; we intentionally retained the frozen template to hold the rendered instructions constant. The quantization files are both IQ4_XS-family but have different per-layer quantization layouts, so this is not a pure fine-tuning ablation.

Thinking text is counted with the same tokenizer as before; the models' vocabulary and merge hashes match. No response hit the 64,000-token ceiling. This single run does not establish timing variability. Article code remains unscored for correctness.

| Suite | Prefill s | Generation s | Total s | Generated tokens | Thinking tokens | Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| quality | 1.17 | 36.28 | 37.5 | 3073 | 2946 | 84.71 |
| speed | 1.91 | 63.64 | 65.63 | 6190 | 4895 | 97.27 |
