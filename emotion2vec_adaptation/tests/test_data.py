"""Regression checks for data integrity, leakage prevention and audio masking."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptation.audio import (AudioDataset, AudioReader, AudioSettings, BatchOrder, ClipBatches, check_durations,
                              collate_audio)
from adaptation.audit import AudioAudit
from adaptation.data import (
    ClipRecord, assert_disjoint_splits, build_labeled_manifest, read_manifest,
    read_unlabeled, select_training_subset, split_records, write_manifest,
)
from utilities.runtime import records_digest
from utilities.generate_demo import create_demo_dataset


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def clip(self, name="clip.wav", seconds=0.3, rate=16000, frequency=220, stereo=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(round(seconds * rate)) / rate
        samples = (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
        if stereo:
            samples = np.stack([samples, samples * 0.5], axis=1)
        sf.write(path, samples, rate)
        return path

    def manifest(self, rows, name="manifest.csv"):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "split", "clip_id"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def records(self, count=20):
        return [ClipRecord(self.clip(f"{i}.wav", frequency=220+i),
                           "happy" if i % 2 else "sad", f"customer-{i}", clip_id=str(i))
                for i in range(count)]

    def test_manifest_paths_are_relative_to_manifest_not_current_directory(self):
        audio = self.clip("audio/一 #.wav")
        manifest = self.manifest([{"path": "audio/一 #.wav", "label": "happy", "group": "call-1"}])
        self.assertEqual(read_manifest(manifest, labels=["happy"])[0].path, audio)

    def test_manifest_rejects_duplicate_paths_invalid_labels_groups_and_splits(self):
        audio = self.clip()
        base = {"path": str(audio), "label": "happy", "group": "call-1"}
        with self.assertRaisesRegex(ValueError, "Duplicate audio"):
            read_manifest(self.manifest([base, base]))
        for override, message in [({"label": "other"}, "configured class"),
                                  ({"group": ""}, "group"), ({"split": "val"}, "split")]:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                read_manifest(self.manifest([{**base, **override}]), labels=["happy"])

    def test_group_split_reproducible_under_row_reordering(self):
        rows = self.records()
        rows.append(ClipRecord(self.clip("same_customer.wav", frequency=700), "happy", rows[0].group))
        first = split_records(rows, seed=13)
        second = split_records(list(reversed(rows)), seed=13)
        self.assertEqual(first, second)
        self.assertEqual(sum(map(len, first.values())), len(rows))
        assert_disjoint_splits(first)
        for name, records in first.items():
            self.assertTrue(all(row.split == name for row in records))

    def test_explicit_split_rejects_partial_rows_and_shared_customer(self):
        rows = self.records(3)
        mixed = [ClipRecord(rows[0].path, "sad", "call-1", "train"), rows[1], rows[2]]
        with self.assertRaisesRegex(ValueError, "Either all rows"):
            split_records(mixed)
        shared = [ClipRecord(row.path, row.label, "shared", name)
                  for row, name in zip(rows, ["train", "validation", "test"])]
        with self.assertRaisesRegex(ValueError, "Split leakage: group"):
            split_records(shared)

    def test_copied_audio_across_groups_is_still_leakage(self):
        source = self.clip()
        copy = self.root / "copy.wav"
        copy.write_bytes(source.read_bytes())
        splits = {"train": [ClipRecord(source, group="call-1")], "test": [ClipRecord(copy, group="call-2")]}
        for audit in (None, AudioAudit(self.root / "audit.jsonl")):
            with self.subTest(audit=audit), self.assertRaisesRegex(ValueError, "byte-identical"):
                assert_disjoint_splits(splits, audit=audit)
        assert_disjoint_splits(splits, check_content=False)  # groups and paths only

    def test_small_group_count_and_invalid_fractions_fail_early(self):
        with self.assertRaisesRegex(ValueError, "Not enough independent"):
            split_records(self.records(2))
        for validation, test in [(0.8, 0.3), (-0.1, 0.2), (float("nan"), 0.1)]:
            with self.subTest(validation=validation), self.assertRaisesRegex(ValueError, "fractions"):
                split_records(self.records(3), validation, test)

    def test_limited_subset_keeps_classes_and_only_uses_supplied_training_rows(self):
        splits = split_records(self.records(), seed=42)
        selected = select_training_subset(splits["train"], 5, ["sad", "happy"], seed=42)
        self.assertEqual(len(selected), 5)
        self.assertEqual({row.label for row in selected}, {"sad", "happy"})
        self.assertTrue(set(selected).issubset(splits["train"]))
        self.assertEqual(selected, select_training_subset(list(reversed(splits["train"])), 5,
                                                        ["sad", "happy"], seed=42))
        with self.assertRaisesRegex(ValueError, "no examples"):
            select_training_subset(splits["train"], 5, ["sad", "happy", "angry"])
        with self.assertRaisesRegex(ValueError, "class count"):
            select_training_subset(splits["train"], 1, ["sad", "happy"])

    def test_unlabeled_excludes_heldout_groups_paths_and_copies(self):
        heldout = self.clip("heldout.wav")
        copied = self.root / "unlabeled" / "unrecognized_call" / "copy.wav"
        copied.parent.mkdir(parents=True)
        copied.write_bytes(heldout.read_bytes())
        self.clip("unlabeled/heldout_customer/other.wav", frequency=600)
        safe = self.clip("unlabeled/train_customer/safe.wav", frequency=700)
        rows = read_unlabeled(self.root / "unlabeled", [ClipRecord(heldout, group="heldout_customer")])
        self.assertEqual([row.path for row in rows], [safe])
        self.assertEqual(rows[0].group, "train_customer")

    def test_manifest_writer_never_overwrites(self):
        target = self.root / "out.csv"
        rows = self.records(2)
        write_manifest(rows, target)
        self.assertEqual(read_manifest(target), rows)
        with self.assertRaises(FileExistsError):
            write_manifest(rows, target)

    def test_manifest_rejects_duplicate_explicit_ids_but_not_duplicate_stems(self):
        first = self.clip("one/audio.wav")
        second = self.clip("two/audio.wav", frequency=600)
        base = [{"path": str(first), "group": "call-1", "label": "happy"},
                {"path": str(second), "group": "call-2", "label": "sad"}]
        self.assertEqual(len(read_manifest(self.manifest(base))), 2)
        with self.assertRaisesRegex(ValueError, "duplicate explicit clip_id"):
            read_manifest(self.manifest([{**row, "clip_id": "same-id"} for row in base]))

    def test_unlabeled_explicit_holdout_never_enters_training(self):
        rows = self.records(3)
        path = self.manifest([{"path": str(row.path), "group": row.group, "split": split}
                              for row, split in zip(rows, ["train", "validation", "test"])])
        self.assertEqual([row.path for row in read_unlabeled(path)], [rows[0].path])
        self.assertEqual(len(read_unlabeled(path, training_only=False)), 3)
        shared = self.manifest([{"path": str(row.path), "group": "shared", "split": split}
                               for row, split in zip(rows, ["train", "validation", "test"])])
        with self.assertRaisesRegex(ValueError, "Split leakage: group"):
            read_unlabeled(shared)

    def votes(self, decisions):
        votes = self.root / "votes.json"
        votes.write_text(json.dumps([
            {"clip_id": clip_id, "decision": decision,
             "status": "bad_clip" if decision == "bad_clip" else "accepted",
             "annotator_id": index}
            for index, (clip_id, decision) in enumerate(decisions)
        ]), encoding="utf-8")
        return votes

    def test_label_join_requires_unanimity_and_excludes_rejected_unknown(self):
        clips = self.manifest([{"path": str(self.clip(f"{i}.wav", frequency=220+i)),
                                "group": f"call-{i}", "clip_id": str(i)} for i in range(4)])
        votes = self.votes([("0", "happy"), ("0", "happy"), ("1", "unknown"), ("2", "bad_clip")])
        target = self.root / "labeled.csv"
        summary = build_labeled_manifest(votes, clips, target, ["happy", "sad"])
        rows = read_manifest(target)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].label, "happy")
        self.assertEqual(summary, {"accepted": 1, "unknown": 1, "bad_clip": 1, "unlabeled": 1})
        conflict = self.votes([("0", "happy"), ("0", "sad")])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            build_labeled_manifest(conflict, clips, self.root / "conflict.csv", ["happy", "sad"])
        self.assertFalse((self.root / "conflict.csv").exists())

    def test_adjudicated_others_class_is_kept_without_relabeling_unknown(self):
        clips = self.manifest([{"path": str(self.clip(f"category-{i}.wav", frequency=300+i)),
                                "group": f"call-{i}", "clip_id": str(i)} for i in range(3)])
        votes = self.votes([("0", "neutral"), ("1", "others"), ("2", "unknown")])
        target = self.root / "with_others.csv"
        summary = build_labeled_manifest(votes, clips, target, ["neutral", "others"])
        rows = read_manifest(target, labels=["neutral", "others"])
        self.assertEqual({row.label for row in rows}, {"neutral", "others"})
        self.assertEqual(summary, {"accepted": 2, "unknown": 1})
        with self.assertRaisesRegex(ValueError, "not in configured classes"):
            build_labeled_manifest(votes, clips, self.root / "missing_others.csv", ["neutral", "sad"])

    def test_audio_resample_stereo_and_padding(self):
        rows = [ClipRecord(self.clip("short.wav", rate=8000, stereo=True), "happy", "c1"),
                ClipRecord(self.clip("long.wav", seconds=0.5, frequency=300), "sad", "c2")]
        dataset = AudioDataset(rows, AudioReader(AudioSettings()), {"happy": 0, "sad": 1})
        batch = collate_audio([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["sources"].shape), (2, 8000))
        self.assertTrue(batch["padding_mask"][0, 4800:].all())
        self.assertFalse(batch["padding_mask"][0, :4800].any())
        self.assertFalse(batch["padding_mask"][1].any())
        self.assertEqual(batch["labels"].tolist(), [0, 1])
        self.assertAlmostEqual(float(batch["sources"][0, :4800].mean()), 0, places=5)
        self.assertTrue(torch.equal(batch["sources"][0, 4800:], torch.zeros(3200)))

    def test_audio_rejects_out_of_range_and_nonfinite_without_trimming(self):
        reader = AudioReader(AudioSettings(max_seconds=0.5))
        for duration in [0.1, 0.6]:
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, "duration"):
                reader.read(self.clip(f"{duration}.wav", seconds=duration))
        path = self.root / "invalid.wav"
        sf.write(path, np.full(5000, np.nan), 16000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "finite"):
            reader.read(path)

    def test_durations_are_checked_up_front_and_every_problem_is_listed(self):
        settings = AudioSettings(min_seconds=0.2, max_seconds=0.5)
        inside = ClipRecord(self.clip("inside.wav", seconds=0.3), group="c1")
        check_durations([inside], settings)
        too_long = ClipRecord(self.clip("too_long.wav", seconds=0.6), group="c2")
        too_short = ClipRecord(self.clip("too_short.wav", seconds=0.1), group="c3")
        broken = self.root / "broken.wav"
        broken.write_bytes(b"not audio")
        for audit in (None, AudioAudit(self.root / "audit.jsonl")):  # headers, or the audit's cached entries
            with self.subTest(audit=audit), self.assertRaisesRegex(ValueError, "3 clip") as raised:
                check_durations([inside, too_long, too_short, ClipRecord(broken, group="c4")], settings, audit)
            for name in ("too_long.wav: 0.60 s", "too_short.wav: 0.10 s", "broken.wav: cannot decode"):
                self.assertIn(name, str(raised.exception))
            self.assertNotIn("inside.wav", str(raised.exception))

    def test_audit_reads_each_file_once_and_rereads_only_changed_files(self):
        paths = [self.clip(f"audit/{i}.wav", seconds=0.2 + 0.1 * i, rate=8000 if i == 2 else 16000) for i in range(3)]
        cache = self.root / "outputs/audit.jsonl"
        first = AudioAudit(cache)
        entries = first.entries(paths)
        self.assertEqual((first.hashed, first.reused), (3, 0))
        self.assertEqual(entries[0]["sha256"], hashlib.sha256(paths[0].read_bytes()).hexdigest())
        self.assertEqual((entries[2]["sample_rate"], entries[2]["frames"]), (8000, 3200))
        self.assertAlmostEqual(entries[1]["duration"], 0.3)
        first.entries(paths)  # already checked in this process: not even stat-ed again
        self.assertEqual((first.hashed, first.reused), (3, 0))
        second = AudioAudit(cache)
        self.assertEqual(second.entries(paths), entries)
        self.assertEqual((second.hashed, second.reused), (0, 3))
        self.clip("audit/0.wav", seconds=0.25)  # new content and size
        self.clip("audit/1.wav", seconds=0.3, frequency=500)  # same size, new content, new mtime
        third = AudioAudit(cache)
        changed = third.entries(paths)
        self.assertEqual((third.hashed, third.reused), (2, 1))
        self.assertNotEqual(changed[0]["sha256"], entries[0]["sha256"])
        self.assertNotEqual(changed[1]["sha256"], entries[1]["sha256"])
        # The documented contract: reuse is keyed by size and mtime_ns, so a same-size rewrite that restores the
        # old mtime is not noticed.
        self.clip("audit/1.wav", seconds=0.3, frequency=220)
        os.utime(paths[1], ns=(changed[1]["mtime_ns"], changed[1]["mtime_ns"]))
        self.assertEqual(AudioAudit(cache).entries([paths[1]])[0]["sha256"], changed[1]["sha256"])
        # An append cut short by a crash costs only that entry; later appends start on a new line.
        with cache.open("ab") as handle:
            handle.write(b'{"path": "/cut-short')
        fourth = AudioAudit(cache)
        self.assertEqual(fourth.ignored_lines, 1)
        fourth.entries([self.clip("audit/new.wav")])
        fifth = AudioAudit(cache)
        fifth.entries(paths + [self.root / "audit/new.wav"])
        self.assertEqual((fifth.ignored_lines, fifth.hashed), (1, 0))
        files = sorted(self.root.rglob("*"))
        memory = AudioAudit()  # no cache file: one read per file in this process only
        memory.entries(paths)
        self.assertEqual(memory.hashed, 3)
        self.assertEqual(sorted(self.root.rglob("*")), files)

    def test_records_digest_follows_content_through_the_audit(self):
        rows = self.records(3)
        audit = AudioAudit(self.root / "audit.jsonl")
        digest = records_digest(rows, audit)
        self.assertEqual(digest, records_digest(rows))  # same value as a fresh in-memory audit
        self.clip("1.wav", frequency=900)
        self.assertNotEqual(digest, records_digest(rows, AudioAudit(self.root / "audit.jsonl")))

    def test_batch_order_is_a_resumable_function_of_seed_and_epoch(self):
        durations = np.random.default_rng(0).uniform(2, 15, 203)
        order = BatchOrder(durations, 4, seed=3, bucket=True, shuffle=True)
        order.set_epoch(2)
        keys = [(epoch, index, clips.tolist()) for epoch, index, clips in order]
        self.assertEqual(len(keys), 51)
        self.assertEqual(sorted(clip for _, _, clips in keys for clip in clips), list(range(203)))
        order.set_epoch(2, start=17)  # a resume: the same batches, the consumed ones skipped
        self.assertEqual([(e, i, c.tolist()) for e, i, c in order], keys[17:])
        self.assertEqual(len(order), 51 - 17)
        order.set_epoch(3)
        self.assertNotEqual([c.tolist() for _, _, c in order], [c for _, _, c in keys])
        plain = BatchOrder(durations, 4, seed=3, bucket=False, shuffle=True)
        plain.set_epoch(2)
        spread = lambda batches: np.mean([np.ptp(durations[c]) for _, _, c in batches])
        self.assertLess(spread(BatchOrder(durations, 4, 3, True, True)), spread(plain) / 10)
        bucketed = BatchOrder(durations, 4, 3, True, True)
        bucketed.set_epoch(2)
        self.assertLess(bucketed.cropped_fraction(), 0.05)
        self.assertGreater(plain.cropped_fraction(), 0.2)
        validation = BatchOrder(durations, 4, seed=3, bucket=True, shuffle=False)
        clips = np.concatenate([c for _, _, c in validation])
        self.assertTrue(np.all(np.diff(durations[clips]) >= 0))  # fixed, sorted batches, keyed as epoch 0
        self.assertEqual({e for e, _, _ in validation}, {0})

    def test_ssl_batches_are_cropped_to_the_shortest_clip_without_padding(self):
        rows = [ClipRecord(self.clip(f"b{i}.wav", seconds=seconds, rate=rate, stereo=rate == 8000), group=f"c{i}")
                for i, (seconds, rate) in enumerate([(0.3, 16000), (0.45, 8000), (0.5, 16000)])]
        reader = AudioReader(AudioSettings())
        batches = ClipBatches(rows, reader, seed=5, crop=True)
        batch = batches[(1, 0, np.array([0, 1, 2]))]
        self.assertEqual(tuple(batch["sources"].shape), (3, 4800))
        self.assertFalse(batch["padding_mask"].any())
        self.assertAlmostEqual(batch["cropped_fraction"], 1 - 3 * 4800 / (4800 + 7200 + 8000), places=6)
        for row, source in zip(rows, batch["sources"]):
            wave = reader.decode(row.path)
            crops = [reader.normalize(wave[offset:offset + 4800], row.path) for offset in range(len(wave) - 4799)]
            self.assertTrue(any(torch.equal(source, crop) for crop in crops))  # re-normalised after the crop
            self.assertAlmostEqual(float(source.mean()), 0, places=5)
        self.assertTrue(torch.equal(batch["sources"], batches[(1, 0, np.array([0, 1, 2]))]["sources"]))
        self.assertFalse(torch.equal(batch["sources"], batches[(2, 0, np.array([0, 1, 2]))]["sources"]))
        padded = ClipBatches(rows, reader, seed=5, crop=False)[(1, 0, np.array([0, 2]))]
        reference = collate_audio([{"source": reader.read(rows[i].path), "label": None, "record": None} for i in (0, 2)])
        self.assertTrue(torch.equal(padded["sources"], reference["sources"]))
        self.assertEqual(padded["cropped_fraction"], 0)

    def test_dataloader_workers_never_change_the_batches(self):
        rows = [ClipRecord(self.clip(f"w{i}.wav", seconds=0.2 + 0.01 * i), group=f"c{i}") for i in range(9)]
        durations = [0.2 + 0.01 * i for i in range(9)]
        batches = ClipBatches(rows, AudioReader(AudioSettings(min_seconds=0.1)), seed=4, crop=True)

        def epochs(workers):
            order = BatchOrder(durations, 2, seed=4, bucket=True, shuffle=True)
            options = {"num_workers": workers, "persistent_workers": True} if workers else {}
            loader = torch.utils.data.DataLoader(batches, sampler=order, batch_size=None,
                                                 generator=torch.Generator().manual_seed(4), **options)
            result = []
            for epoch in (1, 2):
                order.set_epoch(epoch, start=epoch - 1)
                result.extend(batch["sources"] for batch in loader)
            return result

        before = torch.get_rng_state().clone()
        serial, parallel = epochs(0), epochs(2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))  # the masks' global RNG is never drawn
        self.assertEqual(len(serial), 5 + 4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(serial, parallel)))

    def test_demo_fixture_is_valid_synthetic_and_exclusive(self):
        directory = create_demo_dataset(self.root / "generated_demo")
        rows = read_manifest(directory / "labeled.csv", labels=["neutral", "happy"])
        splits = split_records(rows)
        self.assertEqual({name: len(values) for name, values in splits.items()},
                         {"train": 4, "validation": 4, "test": 4})
        unlabeled = read_unlabeled(directory / "unlabeled.csv", exclude_records=rows)
        self.assertEqual(len(unlabeled), 4)
        for row in rows + unlabeled:
            info = sf.info(str(row.path))
            self.assertEqual((info.samplerate, info.channels, info.frames), (16000, 1, 48000))
        summary = build_labeled_manifest(directory / "normalized.json", directory / "clips.csv",
                                         directory / "joined.csv", ["neutral", "happy"])
        self.assertEqual(summary, {"accepted": 12})
        self.assertEqual(read_manifest(directory / "joined.csv"), rows)
        self.assertTrue(json.loads((directory / "demo_manifest.json").read_text())["synthetic"])
        with self.assertRaises(FileExistsError):
            create_demo_dataset(directory)


if __name__ == "__main__":
    unittest.main()
