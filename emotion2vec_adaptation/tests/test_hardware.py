"""Device policy checks plus real CUDA kernels when CUDA is actually available."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.hardware import (cuda_architecture_compatibility, device_environment, memory_metrics,
                                 resolve_precision, training_autocast, validate_device)


class HardwarePolicyTests(unittest.TestCase):
    def test_cpu_auto_is_fp32_and_real_kernels_backward(self):
        self.assertEqual(resolve_precision("cpu", "auto"), "float32")
        result = validate_device("cpu", "auto", run_kernels=True)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["resolved_precision"], "float32")
        self.assertTrue(result["kernel_checks"]["finite_nonzero_backward_gradients"])
        self.assertEqual(result["kernel_checks"]["conv1d_output_dtype"], "torch.float32")

    def test_explicit_cuda_never_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested but unavailable"):
                validate_device("cuda", "auto")
            with self.assertRaisesRegex(RuntimeError, "CUDA requested but unavailable"):
                resolve_precision("cuda", "float32")

    def test_explicit_bf16_requires_cuda_and_fp16_is_not_silently_accepted(self):
        for device in ("cpu", "mps"):
            with self.assertRaisesRegex(ValueError, "only on a CUDA"):
                training_autocast(device, "bfloat16")
        with self.assertRaisesRegex(ValueError, "precision must be"):
            resolve_precision("cpu", "float16")

    def test_cuda_auto_selects_supported_precision(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device", side_effect=lambda device: nullcontext()), \
                patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertEqual(resolve_precision("cuda", "auto"), "bfloat16")
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device", side_effect=lambda device: nullcontext()), \
                patch("torch.cuda.is_bf16_supported", return_value=False):
            self.assertEqual(resolve_precision("cuda", "auto"), "float32")
            with self.assertRaisesRegex(RuntimeError, "does not support BF16"):
                resolve_precision("cuda", "bfloat16")

    def test_blackwell_native_and_generic_ptx_compatibility(self):
        native = cuda_architecture_compatibility((10, 0), ["sm_90", "sm_100", "sm_120"])
        self.assertTrue(native["compatible"])
        self.assertEqual(native["matching_native"], ["sm_100"])
        self.assertTrue(cuda_architecture_compatibility((10, 0), ["compute_90"])["compatible"])
        for architectures in (["sm_90"], ["sm_120", "compute_120"], ["compute_90a"], []):
            self.assertFalse(cuda_architecture_compatibility((10, 0), architectures)["compatible"])

    def test_native_minor_architecture_cannot_run_on_older_minor_device(self):
        self.assertTrue(cuda_architecture_compatibility((8, 6), ["sm_80"])["compatible"])
        self.assertFalse(cuda_architecture_compatibility((8, 0), ["sm_86"])["compatible"])

    def test_blackwell_rejects_benchmark_cuda118_before_allocating(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(10, 0)), \
                patch.object(torch.version, "cuda", "11.8"):
            with self.assertRaisesRegex(RuntimeError, "requirements-llm-cuda128.txt"):
                validate_device("cuda", "float32", run_kernels=False)

    def test_blackwell_valid_metadata_does_not_claim_kernel_test(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(10, 0)), \
                patch("torch.cuda.get_arch_list", return_value=["sm_90", "sm_100"]), \
                patch.object(torch.version, "cuda", "12.8"), \
                patch("utilities.hardware._kernel_check", side_effect=AssertionError("must not run")):
            result = validate_device("cuda", "float32", run_kernels=False)
            self.assertFalse(result["kernel_checks_run"])
            self.assertNotIn("kernel_checks", result)
            self.assertEqual(result["architecture_check"]["target"], "sm_100")

    def test_inventory_is_json_safe_and_identifies_actual_python(self):
        inventory = device_environment("cpu")
        json.dumps(inventory, allow_nan=False)
        self.assertEqual(inventory["selected_device"], "cpu")
        self.assertEqual(inventory["python_executable"], sys.executable)
        self.assertIn("captured_at_utc", inventory)

    def test_inventory_identifies_selected_gpu_on_a_mixed_server(self):
        devices = [SimpleNamespace(name="NVIDIA A100", major=8, minor=0, total_memory=40_000),
                   SimpleNamespace(name="NVIDIA B200", major=10, minor=0, total_memory=180_000)]
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=2), \
                patch("torch.cuda.get_arch_list", return_value=["sm_80", "sm_100"]), \
                patch("torch.cuda.get_device_properties", side_effect=lambda index: devices[index]), \
                patch("utilities.hardware.shutil.which", return_value=None):
            inventory = device_environment("cuda:0")
        self.assertEqual(inventory["selected_cuda_device"]["name"], "NVIDIA A100")
        self.assertEqual(inventory["cuda_devices"][1]["name"], "NVIDIA B200")

    def test_b200_stack_matches_the_mac_versions_with_cuda_128_builds(self):
        """This project uses the benchmark's environment; its B200 file must pin the Mac versions as +cu128 builds."""
        benchmark = Path(__file__).resolve().parents[2] / "emotion_recognition_benchmark"
        def pins(name):
            lines = (benchmark / name).read_text().splitlines()
            return [line.split("#")[0].strip() for line in lines if line.split("#")[0].strip() and not line.startswith("--")]
        self.assertEqual(pins("requirements-llm-cuda128.txt"),
                         [f"{pin}+cu128" for pin in pins("requirements-llm-mac.txt")])
        common = pins("requirements.txt")
        for package in ("funasr==", "transformers==", "numpy==", "pyyaml==", "soundfile==", "matplotlib==", "tqdm=="):
            self.assertTrue(any(pin.startswith(package) for pin in common), package)

    def test_memory_metrics_report_process_peak_on_every_device(self):
        metrics = memory_metrics(torch.device("cpu"))
        self.assertEqual(set(metrics), {"process_peak_rss_gib"})
        self.assertTrue(0.01 < metrics["process_peak_rss_gib"] < 1024)  # GiB on both macOS (bytes) and Linux (KiB)

    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS device in this environment")
    def test_memory_metrics_separate_mps_tensors_from_driver_memory(self):
        tensor = torch.ones(256, 1024, device="mps")  # 1 MiB
        metrics = memory_metrics(torch.device("mps"))
        self.assertGreater(metrics["mps_allocated_gib"], 0)
        self.assertGreaterEqual(metrics["mps_driver_gib"], metrics["mps_allocated_gib"])
        del tensor


@unittest.skipUnless(torch.cuda.is_available(), "No real CUDA device in this environment")
class RealCudaTests(unittest.TestCase):
    def test_memory_metrics_separate_allocated_reserved_and_peak(self):
        tensor = torch.ones(256, 1024, device="cuda")
        metrics = memory_metrics(torch.device("cuda"))
        self.assertGreater(metrics["cuda_allocated_gib"], 0)
        self.assertGreaterEqual(metrics["cuda_reserved_gib"], metrics["cuda_allocated_gib"])
        self.assertGreaterEqual(metrics["cuda_peak_allocated_gib"], metrics["cuda_allocated_gib"])
        del tensor

    def test_actual_cuda_fp32_matmul_conv_and_backward(self):
        result = validate_device("cuda", "float32", run_kernels=True)
        self.assertTrue(result["kernel_checks"]["passed"])

    def test_actual_cuda_bfloat16_matmul_conv_and_backward(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("Selected CUDA GPU does not support BF16")
        result = validate_device("cuda", "bfloat16", run_kernels=True)
        self.assertEqual(result["kernel_checks"]["conv1d_output_dtype"], "torch.bfloat16")

    def test_native_funasr_bfloat16_ssl_backward_and_teacher_ema(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("Selected CUDA GPU does not support BF16")
        from test_models import tiny_backbone
        from adaptation.ssl import MaskedTeacherObjective

        validate_device("cuda", "bfloat16", run_kernels=False)
        device = torch.device("cuda")
        objective = MaskedTeacherObjective(
            tiny_backbone().to(device), target_top_layers=2, predictor_hidden=16,
            freeze_frontend=True, utterance_weight=0.1,
        ).train()
        trainable = [parameter for parameter in objective.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=0.001)
        sources = torch.randn(2, 128, device=device)
        padding = torch.zeros_like(sources, dtype=torch.bool)
        padding[1, 100:] = True
        sources[padding] = 0
        conv_dtypes = []
        local_conv = objective.student.model.modality_encoders["AUDIO"].local_encoder.conv_layers[0][0]
        handle = local_conv.register_forward_hook(lambda module, inputs, output: conv_dtypes.append(output.dtype))
        try:
            with training_autocast(device, "bfloat16"):
                loss, stats = objective(sources, padding)
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(stats["utterance_loss"], 0)
            loss.backward()
        finally:
            handle.remove()
        self.assertIn(torch.bfloat16, conv_dtypes)
        self.assertFalse(objective.teacher.training)
        self.assertTrue(all(parameter.grad is None for parameter in objective.teacher.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in
                            objective.student.model.modality_encoders["AUDIO"].local_encoder.parameters()))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in trainable))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in objective.student.model.blocks.parameters()))
        student_parameter = next(objective.student.model.blocks.parameters())
        teacher_parameter = next(objective.teacher.model.blocks.parameters())
        old_teacher = teacher_parameter.detach().clone()
        optimizer.step()
        expected = old_teacher * 0.5 + student_parameter.detach() * 0.5
        objective.update_teacher(decay=0.5)
        torch.testing.assert_close(teacher_parameter, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(student_parameter.dtype, torch.float32)
        self.assertEqual(teacher_parameter.dtype, torch.float32)
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    unittest.main()
