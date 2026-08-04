import tempfile
import unittest
from pathlib import Path

from benchmark_kimi_k3_deployments import (
    DeviceEndpoint,
    Deployment,
    build_deployments,
    deployment_environment,
    discover_model_files,
    extract_nvme_telemetry,
    parse_device_endpoint,
    server_command,
)


class KimiDeploymentBenchmarkTests(unittest.TestCase):
    def test_builds_capacity_safe_default_profiles(self):
        strix = DeviceEndpoint("hip", 0)
        rtx = DeviceEndpoint("cuda", 0)
        profiles = build_deployments(["primary-only", "heterogeneous"], strix, rtx)
        self.assertEqual(
            profiles,
            [
                Deployment("primary-only-ssd", strix, None),
                Deployment("heterogeneous-ssd", strix, rtx),
            ],
        )

    def test_backend_qualified_endpoints_allow_matching_indices(self):
        strix = parse_device_endpoint("hip:0")
        rtx = parse_device_endpoint("CUDA:0")
        profiles = build_deployments(["heterogeneous"], strix, rtx)
        self.assertEqual(
            profiles, [Deployment("heterogeneous-ssd", strix, rtx)]
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_deployments(["heterogeneous"], strix, strix)
        with self.assertRaisesRegex(ValueError, "secondary-device"):
            build_deployments(["heterogeneous"], strix, None)

    def test_environment_removes_stale_tuning(self):
        base = {
            "PATH": "/bin",
            "DFLASH_MOE_STORAGE": "resident",
            "DFLASH_MOE_NVME_SLOTS": "64",
            "DFLASH_MOE_TP_GPU": "9",
            "DFLASH_MOE_TP_BACKEND": "hip",
            "DFLASH_MOE_PLACEMENT": "/stale.json",
        }
        env = deployment_environment(
            base,
            Deployment(
                "heterogeneous-ssd",
                DeviceEndpoint("hip", 0),
                DeviceEndpoint("cuda", 0),
            ),
            "uring",
            600,
            Path("/new.json"),
            4096,
            True,
            "off",
            16,
            "off",
        )
        self.assertEqual(env["PATH"], "/bin")
        self.assertNotIn("DFLASH_MOE_STORAGE", env)
        self.assertEqual(env["DFLASH_MOE_NVME_SLOTS"], "16")
        self.assertEqual(env["DFLASH_MOE_NVME_BACKEND"], "uring")
        self.assertEqual(env["DFLASH_MOE_NVME_DIRECT"], "off")
        self.assertEqual(env["DFLASH_MOE_NVME_CACHE_FIRST"], "0")
        self.assertEqual(env["DFLASH_MOE_NVME_DEVICE_CACHE_MB"], "4096")
        self.assertEqual(env["DFLASH_MOE_TP_BACKEND"], "cuda")
        self.assertEqual(env["DFLASH_MOE_TP_GPU"], "0")
        self.assertEqual(env["DFLASH_MOE_PRIMARY_SHARE_PER_MILLE"], "600")
        self.assertEqual(env["DFLASH_MOE_PLACEMENT"], "/new.json")
        self.assertEqual(env["DFLASH_MOE_DUAL_STREAM_TRACE"], "1")

    def test_strix_only_has_no_secondary_owner(self):
        env = deployment_environment(
            {"DFLASH_MOE_TP_GPU": "0", "DFLASH_MOE_TP_BACKEND": "cuda"},
            Deployment("primary-only-ssd", DeviceEndpoint("hip", 0), None),
            "auto",
            500,
            None,
            None,
            False,
        )
        self.assertNotIn("DFLASH_MOE_TP_GPU", env)
        self.assertNotIn("DFLASH_MOE_TP_BACKEND", env)
        self.assertEqual(env["DFLASH_MOE_NVME_DIRECT"], "auto")
        self.assertEqual(env["DFLASH_MOE_NVME_CACHE_FIRST"], "1")
        command = server_command(
            Path("server"),
            Path("model.gguf"),
            Deployment("strix", DeviceEndpoint("hip", 0), None),
            8080,
            8192,
            [],
        )
        self.assertIn("hip:0", command)
        self.assertEqual(command[command.index("--moe-storage") + 1], "ssd")
        self.assertEqual(command[command.index("--prefix-cache-slots") + 1], "0")

    def test_discovers_complete_split_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"model-{index:05d}-of-00003.gguf" for index in range(1, 4)]
            for index, path in enumerate(paths, 1):
                path.write_bytes(bytes(index))
            self.assertEqual(discover_model_files(paths[0]), paths)
            with self.assertRaisesRegex(ValueError, "first"):
                discover_model_files(paths[1])
            paths[-1].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                discover_model_files(paths[0])

    def test_extracts_each_owner_telemetry_line(self):
        line = (
            "[moe-nvme] io=io_uring requests=10 reads=9 payload=1.250 GiB "
            "physical=1.500 GiB active-io-rate=3.750 GiB/s cache-hit=10.0% "
            "mean-demand-wait=2.500 ms dedupe=0 upgrades=0 dropped-prefetch=0 "
            "timeouts=0 errors=0 device-cache=1024.0 MiB slots=2 hits=1 misses=9 "
            "evictions=3 graphs=1 graph-hits=8 graph-evictions=0 launches=9\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(line + line)
            rows = extract_nvme_telemetry(log)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["io"], "io_uring")
        self.assertEqual(rows[0]["active_io_gib_s"], 3.75)
        self.assertEqual(rows[0]["errors"], 0)

    def test_telemetry_parser_tolerates_new_and_reordered_fields(self):
        line = (
            "[moe-nvme] io=io_uring+direct launches=107638 requests=87143 "
            "payload=261.811 GiB cache-hit=7.5% device-cache=33444.1 MiB "
            "slots=5436 pinned=0 hits=65354 misses=43574 evictions=38138 "
            "fused-decode-launches=86 fused-decode-experts=1376 future-counter=7\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(line)
            rows = extract_nvme_telemetry(log)
        self.assertEqual(rows[0]["io"], "io_uring+direct")
        self.assertEqual(rows[0]["device_hits"], 65354)
        self.assertEqual(rows[0]["pinned"], 0)
        self.assertEqual(rows[0]["fused_decode_launches"], 86)
        self.assertEqual(rows[0]["future_counter"], 7)


if __name__ == "__main__":
    unittest.main()
