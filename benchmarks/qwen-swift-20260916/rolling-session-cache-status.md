## September 15 hybrid update

The hybrid implementation is installed and Lucebox has been restarted. Exact remains the default; Pi `/cache auto` enables automatic policy per chat after `/reload`. See `cache-optimization-results.md` for current measurements, tradeoffs, and rollback locations. The older status below is historical.

# Lucebox cache: current status

September 15: production restarted and verified healthy on the default Qwen profile, with one stream and GPU prefix checkpoints.

Local Pi now includes **Qwen3.8-27B PFlash (lossy)** as an optional model. Its 8K-window scoring reduced the synthetic cold 80K request from 185.74 to 49.45 seconds. It removes prompt content and is not universally faster: repeated 80K took 50.37 seconds, compared with 1.80 seconds using the exact cache.

Both production profiles passed. Switching back to dense reused 3,584 tokens and answered in 0.90 seconds. Existing shared prefix slots remain; protected per-chat 80K checkpoints are not active in this single-stream path.

See `cache-optimization-results.md` for measured results, limitations and rollback.
