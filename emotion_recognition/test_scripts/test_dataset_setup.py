"""Offline dataset-submodule bootstrap tests using temporary Git repositories and tiny files."""
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/common/setup_datasets.py'
DOWNLOAD_SOURCE = SOURCE.with_name('download_data.py')
GIT = ['git', '-c', 'protocol.file.allow=always', '-c', 'commit.gpgsign=false']
spec = importlib.util.spec_from_file_location('tested_dataset_setup', SOURCE)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
PAYLOAD = b'verified mirrored dataset file\x00\x01'
MANIFEST = {'schema_version': 1, 'repository': 'fixture/data', 'release_tag': 'fixture-v1', 'datasets': {
    'toy': {'storage': 'release', 'name': 'Toy', 'files': [
        {'path': 'audio/one.wav', 'size': len(PAYLOAD), 'sha256': hashlib.sha256(PAYLOAD).hexdigest()}]},
    'crema_d': {'storage': 'external', 'name': 'External fixture', 'source': 'https://example.invalid', 'files': []}}}


def git(where, *args):
    return subprocess.check_output([*GIT, '-C', str(where), *args], stderr=subprocess.PIPE).decode().strip()


def init_repo(path):
    path.mkdir(parents=True)
    git(path, 'init', '-b', 'main')
    git(path, 'config', 'user.name', 'Dataset Bootstrap Fixture')
    git(path, 'config', 'user.email', 'fixture@example.invalid')


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dataset-setup-test-')
        self.root = Path(self.tmp.name)
        self.origin = self.root / 'data-origin'
        init_repo(self.origin)
        (self.origin / 'manifest.json').write_text(json.dumps(MANIFEST))
        (self.origin / 'download.py').write_text('import json,sys\nfrom pathlib import Path\nPath(__file__).with_name("invoked.json").write_text(json.dumps(sys.argv[1:]))\n')
        (self.origin / 'samples').mkdir()
        (self.origin / 'samples/test.wav').write_bytes(b'tiny tracked sample')
        git(self.origin, 'add', '.')
        git(self.origin, 'commit', '-m', 'Dataset metadata fixture')
        self.main = self.root / 'parent'
        init_repo(self.main)
        (self.main / 'emotion_recognition').mkdir()
        git(self.main, 'submodule', 'add', '--name', 'emotion-datasets', str(self.origin), 'emotion_recognition/data')
        git(self.main, 'config', '--file', '.gitmodules', 'submodule.unrelated.path', 'emotion_recognition/models')
        git(self.main, 'config', '--file', '.gitmodules', 'submodule.unrelated.url', str(self.root / 'missing-model-origin'))
        sha = git(self.origin, 'rev-parse', 'HEAD')
        git(self.main, 'update-index', '--add', '--cacheinfo', f'160000,{sha},emotion_recognition/models')
        git(self.main, 'add', '.gitmodules')
        git(self.main, 'commit', '-m', 'Pinned data and unrelated model fixture')
        self.clone = self.root / 'clone'
        subprocess.run([*GIT, 'clone', str(self.main), str(self.clone)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.project = self.clone / 'emotion_recognition'
        self.data = self.project / 'data'

    def tearDown(self):
        self.tmp.cleanup()

    def ensure(self):
        return setup.ensure_submodule(self.project, git_command=GIT)

    def legacy(self, payload=PAYLOAD):
        (self.data / 'toy/audio').mkdir(parents=True, exist_ok=True)
        (self.data / 'toy/audio/one.wav').write_bytes(payload)
        (self.data / 'toy/private-extra.txt').write_text('keep this extra material in backup only')
        (self.data / 'unknown-folder').mkdir()
        (self.data / 'unknown-folder/private.txt').write_text('not public')
        for key in ['crema_d', 'esd']:
            (self.data / key / 'audio').mkdir(parents=True)
            (self.data / key / 'audio/local.wav').write_bytes(b'keep external files')
        (self.data / 'crema_d/.gitattributes').write_text('*.wav filter=lfs\n')

    def test_fresh_clone_initializes_only_data_and_includes_samples(self):
        data = self.ensure()
        self.assertTrue((data / '.git').is_file())
        self.assertEqual((data / 'samples/test.wav').read_bytes(), b'tiny tracked sample')
        self.assertFalse((self.project / 'models/.git').exists())
        self.assertEqual(list(self.project.glob('data.before-submodule-*')), [])

    def test_legacy_files_are_verified_and_reused_without_discarding_extras(self):
        self.legacy()
        data = self.ensure()
        backup = next(self.project.glob('data.before-submodule-*'))
        self.assertEqual(setup.reuse_files(data, MANIFEST, ['toy']), 1)
        self.assertEqual((data / 'toy/audio/one.wav').read_bytes(), PAYLOAD)
        self.assertEqual((backup / 'toy/audio/one.wav').read_bytes(), PAYLOAD)
        self.assertEqual((backup / 'toy/private-extra.txt').read_text(), 'keep this extra material in backup only')
        self.assertFalse((data / 'toy/private-extra.txt').exists())
        self.assertTrue((backup / 'unknown-folder/private.txt').is_file())
        self.assertEqual((backup / 'toy/audio/one.wav').stat().st_ino, (data / 'toy/audio/one.wav').stat().st_ino)

    def test_external_and_archived_files_remain_accessible_without_filter_metadata(self):
        self.legacy()
        data = self.ensure()
        self.assertEqual(setup.preserve_external_files(data), 2)
        for key in ['crema_d', 'esd']:
            self.assertEqual((data / key / 'audio/local.wav').read_bytes(), b'keep external files')
        self.assertFalse((data / 'crema_d/.gitattributes').exists())
        backup = next(self.project.glob('data.before-submodule-*'))
        self.assertTrue((backup / 'crema_d/.gitattributes').is_file())
        self.assertEqual(setup.preserve_external_files(data), 0)

    def test_init_failure_keeps_complete_backup_and_reports_location(self):
        self.legacy()
        git(self.clone, 'config', '--file', '.gitmodules', 'submodule.emotion-datasets.url', str(self.root / 'absent-data'))
        with self.assertRaises(RuntimeError) as caught:
            self.ensure()
        backup = next(self.project.glob('data.before-submodule-*'))
        self.assertIn(str(backup), str(caught.exception))
        self.assertEqual((backup / 'toy/audio/one.wav').read_bytes(), PAYLOAD)
        self.assertTrue((backup / 'esd/audio/local.wav').is_file())

    def test_bad_checksum_is_not_reused(self):
        self.legacy(b'x' * len(PAYLOAD))
        data = self.ensure()
        self.assertEqual(setup.reuse_files(data, MANIFEST, ['toy']), 0)
        self.assertFalse((data / 'toy/audio/one.wav').exists())
        backup = next(self.project.glob('data.before-submodule-*'))
        self.assertEqual((backup / 'toy/audio/one.wav').read_bytes(), b'x' * len(PAYLOAD))

    def test_cross_device_copy_keeps_original_and_verifies_copy(self):
        self.legacy()
        data = self.ensure()
        with mock.patch.object(setup.os, 'link', side_effect=OSError(errno.EXDEV, 'fixture cross-device')):
            self.assertEqual(setup.reuse_files(data, MANIFEST, ['toy']), 1)
        backup = next(self.project.glob('data.before-submodule-*'))
        self.assertEqual((backup / 'toy/audio/one.wav').read_bytes(), PAYLOAD)
        self.assertEqual((data / 'toy/audio/one.wav').read_bytes(), PAYLOAD)
        self.assertNotEqual((backup / 'toy/audio/one.wav').stat().st_ino, (data / 'toy/audio/one.wav').stat().st_ino)

    def test_target_is_never_overwritten_by_local_reuse(self):
        self.legacy()
        data = self.ensure()
        (data / 'toy/audio').mkdir(parents=True)
        (data / 'toy/audio/one.wav').write_bytes(b'existing target')
        self.assertEqual(setup.reuse_files(data, MANIFEST, ['toy']), 0)
        self.assertEqual((data / 'toy/audio/one.wav').read_bytes(), b'existing target')

    def test_paths_and_escaping_symlinks_are_rejected(self):
        data = self.ensure()
        malformed = json.loads(json.dumps(MANIFEST))
        malformed['datasets']['toy']['files'][0]['path'] = '../outside.wav'
        with self.assertRaises(ValueError):
            setup.selected_datasets(malformed, ['toy'])
        outside = self.root / 'outside'
        outside.mkdir()
        (data / 'toy').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            setup.reuse_files(data, MANIFEST, ['toy'])
        self.assertFalse((outside / 'audio/one.wav').exists())

    def test_cli_forwards_models_ca_path_spaces_and_verification(self):
        self.ensure()
        script = self.project / 'scripts/common/setup_datasets.py'
        script.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE, script)
        proc = subprocess.run([sys.executable, str(script), 'toy', '--verify-only', '--workers', '3',
                               '--ca-bundle', '/fixture/company ca.pem'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads((self.data / 'invoked.json').read_text()),
                         ['toy', '--workers', '3', '--verify-only', '--ca-bundle', '/fixture/company ca.pem'])

    def test_default_selection_excludes_external_and_all_lists_them(self):
        self.assertEqual(setup.selected_datasets(MANIFEST), ['toy'])
        self.assertEqual(setup.selected_datasets(MANIFEST, ['mirrors']), ['toy'])
        self.assertEqual(setup.selected_datasets(MANIFEST, listing=True), ['toy', 'crema_d'])
        self.assertEqual(setup.selected_datasets(MANIFEST, ['all']), ['toy', 'crema_d'])
        with mock.patch.object(setup, 'ensure_submodule', return_value=self.data), \
                mock.patch.object(setup, 'preserve_external_files') as preserve, \
                mock.patch.object(setup.subprocess, 'call') as download:
            self.data.mkdir(exist_ok=True)
            (self.data / 'manifest.json').write_text(json.dumps(MANIFEST))
            self.assertEqual(setup.main(['crema_d']), 1)
            preserve.assert_called_once()
            download.assert_not_called()

    def test_empty_mirror_selection_never_downloads_all(self):
        self.ensure()
        manifest = dict(MANIFEST, datasets={'crema_d': MANIFEST['datasets']['crema_d']})
        (self.data / 'manifest.json').write_text(json.dumps(manifest))
        with mock.patch.object(setup, 'ensure_submodule', return_value=self.data), \
                mock.patch.object(setup.subprocess, 'call') as download:
            self.assertEqual(setup.main([]), 0)
            download.assert_not_called()


class LegacyGitRecoveryTests(unittest.TestCase):
    """A real checkout removes tracked WAV/parts before bootstrap restores their local Git blobs."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dataset-git-recovery-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.main = self.root / 'history-parent'
        init_repo(self.main)
        old = self.main / 'emotion_recognition/data'
        (old / 'toy/audio').mkdir(parents=True)
        (old / 'toy/audio/one.wav').write_bytes(PAYLOAD)
        (old / 'toy/private-extra.txt').write_bytes(b'not in the mirrored whitelist')
        (old / 'toy/audio/link.wav').symlink_to('one.wav')
        self.parquet = b'PAR1' + b'parquet-bytes\x00' * 5 + b'PAR1'
        (old / 'nemo/data').mkdir(parents=True)
        (old / 'nemo/data/train.parquet.part000').write_bytes(self.parquet[:31])
        (old / 'nemo/data/train.parquet.part001').write_bytes(self.parquet[31:])
        git(self.main, 'add', 'emotion_recognition/data')
        git(self.main, 'commit', '-m', 'Old tracked WAV and Parquet transfer parts')
        self.legacy_commit = git(self.main, 'rev-parse', 'HEAD')
        self.manifest = json.loads(json.dumps(MANIFEST))
        self.manifest['legacy_git'] = {'commit': self.legacy_commit, 'prefix': 'emotion_recognition/data'}
        self.manifest['datasets']['nemo'] = {'name': 'nEMO fixture', 'storage': 'release', 'files': [
            {'path': 'data/train.parquet', 'size': len(self.parquet), 'sha256': hashlib.sha256(self.parquet).hexdigest()}]}
        self.clone = self.root / 'old-computer'
        subprocess.run([*GIT, 'clone', str(self.main), str(self.clone)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertTrue((self.clone / 'emotion_recognition/data/toy/audio/one.wav').exists())
        origin = self.root / 'data-origin'
        init_repo(origin)
        (origin / 'manifest.json').write_text(json.dumps(self.manifest))
        (origin / 'download.py').write_text('raise RuntimeError("Network downloader must not run in this fixture")\n')
        git(origin, 'add', '.')
        git(origin, 'commit', '-m', 'Data metadata fixture')
        git(self.main, 'rm', '-r', 'emotion_recognition/data')
        git(self.main, 'submodule', 'add', '--name', 'emotion-datasets', str(origin), 'emotion_recognition/data')
        git(self.main, 'commit', '-am', 'Migrate tracked data to metadata submodule')
        git(self.clone, 'fetch', 'origin')
        git(self.clone, 'merge', '--ff-only', 'origin/main')
        self.project = self.clone / 'emotion_recognition'
        self.data = self.project / 'data'
        self.assertFalse((self.data / 'toy/audio/one.wav').exists())
        self.assertFalse((self.data / 'nemo/data/train.parquet.part000').exists())
        setup.ensure_submodule(self.project, git_command=GIT)
        self.assertEqual(list(self.project.glob('data.before-submodule-*')), [])

    def test_checkout_deleted_wav_and_parquet_parts_recover_through_one_batch(self):
        original = subprocess.Popen
        with mock.patch.object(setup.subprocess, 'Popen', wraps=original) as opened:
            count = setup.reuse_legacy_git(self.data, self.manifest, ['toy', 'nemo'], git_command=GIT)
        self.assertEqual(count, 2)
        self.assertEqual((self.data / 'toy/audio/one.wav').read_bytes(), PAYLOAD)
        self.assertEqual((self.data / 'nemo/data/train.parquet').read_bytes(), self.parquet)
        batches = [call.args[0] for call in opened.call_args_list if '--batch' in call.args[0]]
        self.assertEqual(len(batches), 1)
        self.assertFalse((self.data / 'toy/private-extra.txt').exists())
        self.assertFalse((self.data / 'nemo/data/train.parquet.part000').exists())
        self.assertEqual(subprocess.check_output([*GIT, '-C', str(self.clone), 'show',
                         self.legacy_commit + ':emotion_recognition/data/toy/audio/one.wav']), PAYLOAD)
        self.assertEqual(setup.reuse_legacy_git(self.data, self.manifest, ['toy', 'nemo'], git_command=GIT), 0)

    def test_bad_size_or_sha_does_not_publish_or_leave_partial_files(self):
        for change in [{'size': len(PAYLOAD) + 1}, {'sha256': '0' * 64}]:
            malformed = json.loads(json.dumps(self.manifest))
            malformed['datasets']['toy']['files'][0].update(change)
            self.assertEqual(setup.reuse_legacy_git(self.data, malformed, ['toy'], git_command=GIT), 0)
            self.assertFalse((self.data / 'toy/audio/one.wav').exists())
            self.assertEqual(list(self.data.rglob('*.legacy-git-*.partial')), [])

    def test_existing_unknown_target_is_not_overwritten(self):
        target = self.data / 'toy/audio/one.wav'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'user local material')
        self.assertEqual(setup.reuse_legacy_git(self.data, self.manifest, ['toy'], git_command=GIT), 0)
        self.assertEqual(target.read_bytes(), b'user local material')

    def test_missing_commit_falls_back_without_any_fetch(self):
        malformed = json.loads(json.dumps(self.manifest))
        malformed['legacy_git']['commit'] = '0' * 40
        original = subprocess.Popen
        with mock.patch.object(setup.subprocess, 'Popen', wraps=original) as opened:
            self.assertEqual(setup.reuse_legacy_git(self.data, malformed, ['toy'], git_command=GIT), 0)
        self.assertFalse(any('fetch' in call.args[0] for call in opened.call_args_list))
        self.assertFalse((self.data / 'toy/audio/one.wav').exists())

    def test_real_shallow_clone_does_not_fetch_old_history(self):
        shallow = self.root / 'shallow-computer'
        subprocess.run([*GIT, 'clone', '--depth', '1', '--no-local', self.main.as_uri(), str(shallow)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        project = shallow / 'emotion_recognition'
        data = setup.ensure_submodule(project, git_command=GIT)
        for attempt in range(2):
            check = subprocess.run([*GIT, '-C', str(shallow), 'cat-file', '-e', self.legacy_commit],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(check.returncode, 0)
            if attempt == 0:
                self.assertEqual(setup.reuse_legacy_git(data, self.manifest, ['toy', 'nemo'], git_command=GIT), 0)
        self.assertFalse((data / 'toy/audio/one.wav').exists())

    def test_missing_parquet_part_does_not_publish_a_truncated_file(self):
        repository, environment, blobs = setup.legacy_blob_map(self.data, self.manifest['legacy_git'], git_command=GIT)
        blobs = dict(blobs)
        del blobs['nemo/data/train.parquet.part000']
        with mock.patch.object(setup, 'legacy_blob_map', return_value=(repository, environment, blobs)):
            self.assertEqual(setup.reuse_legacy_git(self.data, self.manifest, ['nemo'], git_command=GIT), 0)
        self.assertFalse((self.data / 'nemo/data/train.parquet').exists())

    def test_symlink_blobs_and_internal_symlink_ancestors_are_rejected(self):
        malformed = json.loads(json.dumps(self.manifest))
        malformed['datasets']['toy']['files'][0]['path'] = 'audio/link.wav'
        self.assertEqual(setup.reuse_legacy_git(self.data, malformed, ['toy'], git_command=GIT), 0)
        self.assertFalse((self.data / 'toy/audio/link.wav').exists())
        (self.data / 'alternate').mkdir()
        (self.data / 'toy/audio').mkdir(parents=True)
        (self.data / 'toy/audio').rmdir()
        (self.data / 'toy/audio').symlink_to(self.data / 'alternate', target_is_directory=True)
        with self.assertRaises(ValueError):
            setup.reuse_legacy_git(self.data, self.manifest, ['toy'], git_command=GIT)
        self.assertEqual(list((self.data / 'alternate').iterdir()), [])


class SourceDispatchTests(unittest.TestCase):
    def load(self):
        spec = importlib.util.spec_from_file_location('tested_data_download', DOWNLOAD_SOURCE)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {'yaml': types.SimpleNamespace(safe_load=json.loads)}):
            spec.loader.exec_module(module)
        self.tmp = tempfile.TemporaryDirectory(prefix='data-dispatch-test-')
        self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name)
        module.ROOT = folder
        module.CONFIG = folder / 'download.json'
        module.CONFIG.write_text(json.dumps({'provider': 'github', 'dataset': 'mirrors', 'workers': 4, 'crema_d': 'samples'}))
        return module, folder

    def test_github_mirrors_dispatches_without_touching_sources(self):
        module, folder = self.load()
        with mock.patch.object(module.subprocess, 'call', return_value=17) as run, \
                mock.patch.object(module, 'configure_https') as configure:
            self.assertEqual(module.main(['--ca-bundle', '/fixture/company ca.pem']), 17)
        self.assertEqual(run.call_args.args[0], [sys.executable, str(folder / 'scripts/common/setup_datasets.py'),
                                               'mirrors', '--workers', '4', '--ca-bundle', '/fixture/company ca.pem'])
        configure.assert_not_called()

    def test_explicit_source_uses_publisher_without_downloading_mirrors(self):
        module, _ = self.load()
        with mock.patch.object(module, 'configure_https') as configure, \
                mock.patch.object(module, 'download_dataset') as download, \
                mock.patch.object(module.subprocess, 'call') as run:
            self.assertEqual(module.main(['--dataset', 'mess', '--source']), 0)
        download.assert_called_once_with('mess', 4, 4)
        configure.assert_called_once_with(None)
        run.assert_not_called()

    def test_all_original_sources_explicitly_excludes_esd(self):
        module, _ = self.load()
        with mock.patch.object(module, 'configure_https'), \
                mock.patch.object(module, 'download_dataset') as download, \
                mock.patch.object(module, 'download_crema_d') as crema:
            self.assertEqual(module.main(['--dataset', 'all', '--source']), 0)
        names = [call.args[0] for call in download.call_args_list]
        self.assertNotIn('esd', names)
        self.assertEqual(len(names), 12)
        crema.assert_called_once_with('full', 4)

    def test_default_six_samples_initialize_metadata_without_release_archive(self):
        module, folder = self.load()
        module.CONFIG.write_text(json.dumps({'provider': 'github', 'dataset': 'crema_d', 'workers': 4, 'crema_d': 'samples'}))
        ensure = mock.Mock(return_value=folder / 'data')
        with mock.patch.dict(sys.modules, {'setup_datasets': types.SimpleNamespace(ensure_submodule=ensure)}), \
                mock.patch.object(module, 'configure_https'), \
                mock.patch.object(module, 'download_crema_d') as crema, \
                mock.patch.object(module.subprocess, 'call') as release:
            self.assertEqual(module.main([]), 0)
        ensure.assert_called_once_with()
        crema.assert_called_once_with('samples', 4)
        release.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
