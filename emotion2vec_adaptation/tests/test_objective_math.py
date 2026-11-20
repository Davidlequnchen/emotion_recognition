"""Independent mathematical checks of masking, stop-gradient and loss support."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptation.ssl import MaskedTeacherObjective
from test_models import tiny_backbone


class ObjectiveMathematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_loss_and_exact_derivative_only_use_masked_valid_frames(self):
        """Independently compute the target and analytic MSE derivative."""
        torch.manual_seed(27)
        objective = MaskedTeacherObjective(tiny_backbone(), target_top_layers=2, predictor_hidden=8).train()
        sources = torch.randn(2, 128)
        padding = torch.zeros_like(sources, dtype=torch.bool)
        padding[1, 96:] = True
        observed = {}
        teacher_frames = objective.teacher.frames
        student_frames = objective.student.frames

        def observe_teacher(*args, **kwargs):
            observed["teacher"] = teacher_frames(*args, **kwargs)
            return observed["teacher"]

        def observe_student(*args, **kwargs):
            observed["student"] = student_frames(*args, **kwargs)
            return observed["student"]

        def observe_predictions(module, inputs, output):
            output.retain_grad()
            observed["predictions"] = output

        handle = objective.predictor.register_forward_hook(observe_predictions)
        self.addCleanup(handle.remove)
        teacher_before = objective.teacher.fingerprint
        with patch.object(objective.teacher, "frames", side_effect=observe_teacher), \
                patch.object(objective.student, "frames", side_effect=observe_student):
            loss, stats = objective(sources, padding)

        teacher = observed["teacher"]
        valid = ~teacher["padding_mask"]
        mask = observed["student"]["mask"].mask.bool()
        self.assertFalse((mask & ~valid).any())
        targets = []
        for layer in teacher["layer_results"][-2:]:
            layer = layer[:, objective.teacher.num_extra_tokens:]
            normalized = torch.zeros_like(layer)
            for index in range(len(sources)):
                values = layer[index, valid[index]]
                normalized[index, valid[index]] = (
                    (values - values.mean(0)) / (values.var(0, unbiased=False) + 1e-5).sqrt()
                )
            targets.append(normalized)
        target = torch.stack(targets).mean(0)
        prediction = observed["predictions"]
        residual = prediction.detach()[mask] - target[mask]
        expected_loss = residual.square().sum() / residual.numel()
        torch.testing.assert_close(loss.detach(), expected_loss)
        self.assertEqual(stats["utterance_loss"], 0.0)
        loss.backward()
        expected_gradient = torch.zeros_like(prediction)
        expected_gradient[mask] = 2 * residual / residual.numel()
        torch.testing.assert_close(prediction.grad, expected_gradient)
        self.assertEqual(objective.teacher.fingerprint, teacher_before)
        self.assertTrue(all(parameter.grad is None for parameter in objective.teacher.parameters()))
        self.assertTrue(all(not layer.requires_grad for layer in teacher["layer_results"]))

    def test_native_student_zeros_masked_latents_before_context_encoding(self):
        """A real FunASR hook verifies mask application, beyond checking its indices."""
        torch.manual_seed(37)
        objective = MaskedTeacherObjective(tiny_backbone(), target_top_layers=2, predictor_hidden=8)
        audio_encoder = objective.student.model.modality_encoders["AUDIO"]
        captured = {}
        native_compute_mask = audio_encoder.compute_mask

        def inspect_mask(*args, **kwargs):
            masked_features, info = native_compute_mask(*args, **kwargs)
            captured["mask"] = info.mask.bool()
            captured["features"] = masked_features.detach()
            return masked_features, info

        with patch.object(audio_encoder, "compute_mask", side_effect=inspect_mask):
            objective(torch.randn(2, 128))
        self.assertGreater(captured["mask"].sum().item(), 0)
        self.assertTrue((captured["features"][captured["mask"]] == 0).all())
        self.assertGreater(captured["features"][~captured["mask"]].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
