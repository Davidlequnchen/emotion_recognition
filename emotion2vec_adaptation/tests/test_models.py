"""Gradient, masking and export invariants using the real tiny FunASR network."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptation.backbone import EmotionBackbone, masked_mean, state_fingerprint
from adaptation.ssl import MaskedTeacherObjective, sample_span_mask, temporal_instance_normalize


def tiny_backbone() -> EmotionBackbone:
    """Use native mask/context behavior with tiny dimensions, no model download."""
    from funasr.models.emotion2vec.model import Emotion2vec

    config_path = Path(__file__).resolve().parents[2] / "emotion_recognition_benchmark/models/emotion2vec_plus_large/config.yaml"
    config = yaml.safe_load(config_path.read_text())["model_conf"]
    config.update(
        embed_dim=16, num_heads=2, depth=2, encoder_dropout=0.0,
        attention_dropout=0.0, post_mlp_drop=0.0, dropout_input=0.0,
    )
    config["modalities"]["audio"].update(
        feature_encoder_spec="[(8, 3, 2), (8, 3, 2)]",
        conv_pos_depth=1, conv_pos_width=5, conv_pos_groups=2,
        prenet_depth=1, model_depth=3, num_alibi_heads=2,
        num_extra_tokens=2, prenet_dropout=0.0, decoder=None,
    )
    return EmotionBackbone(Emotion2vec(model_conf=config, vocab_size=-1))


class MaskingTests(unittest.TestCase):
    def test_mask_never_selects_padding_or_every_valid_frame(self):
        valid = torch.arange(20)[None, :] < torch.tensor([20, 11, 2])[:, None]
        for span in (1, 5, 100):
            mask = sample_span_mask(valid, 0.55, span)
            self.assertFalse((mask & ~valid).any())
            self.assertTrue(mask.any(dim=1).all())
            self.assertTrue((valid & ~mask).any(dim=1).all())

    def test_one_frame_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            sample_span_mask(torch.tensor([[True, False]]), 0.5, 2)

    def test_padding_excluded_from_pooling_and_target_normalization(self):
        features = torch.tensor([[[1.0, 4.0], [3.0, 8.0], [9e6, -9e6]]])
        valid = torch.tensor([[True, True, False]])
        torch.testing.assert_close(masked_mean(features, valid), torch.tensor([[2.0, 6.0]]))
        norm = temporal_instance_normalize(features, valid)
        torch.testing.assert_close(norm[:, :2], temporal_instance_normalize(features[:, :2], valid[:, :2]))
        self.assertTrue((norm[:, 2] == 0).all())


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_masked_native_gradient_teacher_and_frontend_are_frozen(self):
        objective = MaskedTeacherObjective(
            tiny_backbone(), target_top_layers=2, predictor_hidden=8,
            utterance_weight=0.25, freeze_frontend=True,
        ).train()
        sources = torch.randn(2, 128)
        padding = torch.zeros_like(sources, dtype=torch.bool)
        padding[1, 100:] = True
        loss, stats = objective(sources, padding)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(stats["masked_frames"], 0)
        self.assertLess(stats["masked_frames"], stats["valid_frames"])
        self.assertGreater(stats["utterance_loss"], 0)
        loss.backward()
        self.assertFalse(objective.teacher.training)
        self.assertTrue(all(p.grad is None for p in objective.teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in objective.student.model.modality_encoders["AUDIO"].local_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in objective.student.model.blocks.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in objective.predictor.parameters()))

    def test_final_target_and_learnability_baselines(self):
        objective = MaskedTeacherObjective(tiny_backbone(), predictor_hidden=8, target="final",
                                           utterance_weight=0.5).train()
        sources = torch.randn(2, 128)
        loss, stats = objective(sources, None)
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(stats["constant_loss"], stats["zero_loss"] + 1e-9)  # the mean is the best constant
        self.assertIn("teacher_pooled_std", stats)
        # final target: each frame layer-normalised, so the zero baseline of the frame part is 1
        frame_only = MaskedTeacherObjective(tiny_backbone(), predictor_hidden=8, target="final")
        self.assertAlmostEqual(frame_only(sources, None)[1]["zero_loss"], 1.0, places=4)
        with self.assertRaisesRegex(ValueError, "both be zero"):
            MaskedTeacherObjective(tiny_backbone(), frame_weight=0.0, utterance_weight=0.0)

    def test_ema_only_changes_after_update_and_matches_decay(self):
        objective = MaskedTeacherObjective(tiny_backbone(), target_top_layers=2, predictor_hidden=8)
        teacher_parameter = next(objective.teacher.parameters())
        student_parameter = next(objective.student.parameters())
        original = teacher_parameter.detach().clone()
        with torch.no_grad():
            student_parameter.add_(4)
        torch.testing.assert_close(teacher_parameter, original)
        objective.update_teacher(decay=0.75)
        torch.testing.assert_close(teacher_parameter, original + 1)
        objective.train(False).train(True)
        self.assertFalse(objective.teacher.training)

    def test_encoder_export_roundtrip_and_fingerprint_detects_change(self):
        original = tiny_backbone().eval()
        fingerprint = original.fingerprint
        state = original.encoder_state_dict()
        self.assertFalse(any(name.startswith("proj.") or ".decoder." in name for name in state))
        replacement = tiny_backbone().eval()
        replacement.load_encoder(state)
        self.assertEqual(replacement.fingerprint, fingerprint)
        sources = torch.randn(1, 128)
        with torch.no_grad():
            torch.testing.assert_close(original.embed(sources), replacement.embed(sources))
            next(replacement.parameters()).add_(1)
        self.assertNotEqual(replacement.fingerprint, fingerprint)
        state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):
            replacement.load_encoder(state)

    def test_layer_embeddings_cover_every_block_and_end_with_the_option_a_embedding(self):
        backbone = tiny_backbone().eval()
        blocks = backbone.blocks()
        self.assertEqual(len(blocks), 3)  # tiny network: 1 prenet block + 2 shared blocks
        sources = torch.randn(2, 128)
        padding = torch.zeros_like(sources, dtype=torch.bool)
        padding[1, 100:] = True
        with torch.no_grad():
            layers = backbone.layer_embeddings(sources, padding)
            self.assertEqual(tuple(layers.shape), (2, 3, 16))
            torch.testing.assert_close(layers[:, -1], backbone.embed(sources, padding))
            self.assertFalse(torch.allclose(layers[:, 0], layers[:, -1]))
        self.assertTrue(all(not block._forward_hooks for block in blocks))  # hooks removed after the call

    def test_embed_each_ignores_batch_partners_while_padded_batches_do_not(self):
        backbone = tiny_backbone().eval()
        short, long = torch.randn(1, 80), torch.randn(1, 128)
        sources = torch.zeros(2, 128)
        sources[0, :80], sources[1] = short[0], long[0]
        padding = torch.zeros_like(sources, dtype=torch.bool)
        padding[0, 80:] = True
        with torch.no_grad():
            alone = backbone.embed(short)
            torch.testing.assert_close(backbone.embed_each(sources, padding)[0:1], alone)
            torch.testing.assert_close(backbone.embed_each(sources, padding, layers=True)[0:1],
                                       backbone.layer_embeddings(short))
            self.assertFalse(torch.allclose(backbone.embed(sources, padding)[0:1], alone, atol=1e-6))

    def test_fingerprint_includes_shape_and_name(self):
        data = torch.ones(4)
        self.assertNotEqual(state_fingerprint({"a": data}), state_fingerprint({"b": data}))
        self.assertNotEqual(state_fingerprint({"a": data}), state_fingerprint({"a": data.reshape(2, 2)}))


if __name__ == "__main__":
    unittest.main()
