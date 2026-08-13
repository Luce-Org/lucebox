# Hardware Qualification

These launchers reproduce performance results for specific hardware profiles.
They are separate from the reusable benchmark clients because they may change
device settings and require machine-specific inputs.

- `deepseek4/qualify_ds4_q5_amd.sh`: R9700 plus Strix Halo q=5 qualification
- `deepseek4/rocprof_server_wrapper.sh`: delayed ROCm profiler launcher
- `deepseek4/analyze_rocprof_overlap.py`: profiler overlap summary
