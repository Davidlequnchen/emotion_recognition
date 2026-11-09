"""Offline model-submodule bootstrap tests using temporary Git repositories."""
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/common/setup_models.py'
DOWNLOAD_SOURCE = SOURCE.with_name('download_model.py')
GIT = ['git', '-c', 'protocol.file.allow=always', '-c', 'commit.gpgsign=false']
spec = importlib.util.spec_from_file_location('tested_setup', SOURCE)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
PAYLOAD = b'verified local test model weights\x00\x01\x02'
MANIFEST = {'schema_version':1,'repository':'fixture/models','release_tag':'fixture-v1','models':{'toy':{
    'files':[{'path':'model.bin','size':len(PAYLOAD),'sha256':hashlib.sha256(PAYLOAD).hexdigest(),
              'parts':[{'asset':'toy--model.bin','size':len(PAYLOAD),'sha256':hashlib.sha256(PAYLOAD).hexdigest()}]}]}}}


def git(where, *args):
    return subprocess.check_output([*GIT,'-C',str(where),*args],stderr=subprocess.PIPE).decode().strip()


def init_repo(path):
    path.mkdir(parents=True)
    git(path,'init','-b','main')
    git(path,'config','user.name','Bootstrap Fixture')
    git(path,'config','user.email','fixture@example.invalid')


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='emotion-setup-test-')
        self.root = Path(self.tmp.name)
        self.origin = self.root/'model-origin'
        init_repo(self.origin)
        (self.origin/'manifest.json').write_text(json.dumps(MANIFEST))
        (self.origin/'download.py').write_text('import json,sys\nfrom pathlib import Path\nPath(__file__).with_name("invoked.json").write_text(json.dumps(sys.argv[1:]))\n')
        (self.origin/'README.md').write_text('Synthetic model fixture\n')
        git(self.origin,'add','manifest.json','download.py','README.md')
        git(self.origin,'commit','-m','Model metadata fixture')
        self.main = self.root/'parent'
        init_repo(self.main)
        (self.main/'emotion_recognition').mkdir()
        git(self.main,'submodule','add','--name','emotion-models',str(self.origin),'emotion_recognition/models')
        # An unrelated tracked submodule deliberately has an inaccessible URL. Bootstrap must not touch it.
        git(self.main,'config','--file','.gitmodules','submodule.unrelated.path','antideepfake/models/voice_clone')
        git(self.main,'config','--file','.gitmodules','submodule.unrelated.url',str(self.root/'missing-unrelated'))
        child_sha=git(self.origin,'rev-parse','HEAD')
        git(self.main,'update-index','--add','--cacheinfo',f'160000,{child_sha},antideepfake/models/voice_clone')
        git(self.main,'add','.gitmodules')
        git(self.main,'commit','-m','Pinned model submodule and unrelated fixture')
        self.clone = self.root/'clone'
        subprocess.run([*GIT,'clone',str(self.main),str(self.clone)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.project=self.clone/'emotion_recognition'
        self.models=self.project/'models'

    def tearDown(self):
        self.tmp.cleanup()

    def ensure(self):
        return setup.ensure_submodule(self.project,git_command=GIT)

    def legacy(self,payload=PAYLOAD):
        (self.models/'toy').mkdir(parents=True,exist_ok=True)
        (self.models/'toy/model.bin').write_bytes(payload)
        (self.models/'unknown-notes.txt').write_text('do not discard')

    def test_fresh_clone_initializes_only_models(self):
        models=self.ensure()
        self.assertTrue((models/'.git').is_file())
        self.assertTrue((models/'manifest.json').is_file())
        self.assertFalse((self.clone/'antideepfake/models/voice_clone/.git').exists())
        self.assertEqual(list(self.project.glob('models.before-submodule-*')),[])

    def test_preserves_legacy_directory_and_reuses_verified_hardlink(self):
        self.legacy()
        models=self.ensure()
        backups=list(self.project.glob('models.before-submodule-*'))
        self.assertEqual(len(backups),1)
        backup=backups[0]
        self.assertEqual((backup/'unknown-notes.txt').read_text(),'do not discard')
        self.assertEqual(setup.reuse_weights(models,MANIFEST,['toy']),1)
        self.assertEqual((models/'toy/model.bin').read_bytes(),PAYLOAD)
        self.assertEqual((backup/'toy/model.bin').read_bytes(),PAYLOAD)
        self.assertEqual((models/'toy/model.bin').stat().st_ino,(backup/'toy/model.bin').stat().st_ino)
        self.assertEqual(setup.reuse_weights(models,MANIFEST,['toy']),0)

    def test_init_failure_keeps_all_legacy_files_and_reports_backup(self):
        self.legacy()
        git(self.clone,'config','--file','.gitmodules','submodule.emotion-models.url',str(self.root/'missing-model-origin'))
        with self.assertRaises(RuntimeError) as caught:self.ensure()
        backups=list(self.project.glob('models.before-submodule-*'))
        self.assertEqual(len(backups),1)
        self.assertIn(str(backups[0]),str(caught.exception))
        self.assertEqual((backups[0]/'toy/model.bin').read_bytes(),PAYLOAD)
        self.assertEqual((backups[0]/'unknown-notes.txt').read_text(),'do not discard')

    def test_bad_local_checksum_is_not_reused(self):
        self.legacy(b'x'*len(PAYLOAD))
        models=self.ensure()
        self.assertEqual(setup.reuse_weights(models,MANIFEST,['toy']),0)
        self.assertFalse((models/'toy/model.bin').exists())
        backup=next(self.project.glob('models.before-submodule-*'))
        self.assertEqual((backup/'toy/model.bin').read_bytes(),b'x'*len(PAYLOAD))

    def test_cross_device_reuse_copies_without_removing_source(self):
        self.legacy()
        models=self.ensure()
        with mock.patch.object(setup.os,'link',side_effect=OSError(errno.EXDEV,'fixture cross-device')):
            self.assertEqual(setup.reuse_weights(models,MANIFEST,['toy']),1)
        backup=next(self.project.glob('models.before-submodule-*'))
        self.assertEqual((backup/'toy/model.bin').read_bytes(),PAYLOAD)
        self.assertEqual((models/'toy/model.bin').read_bytes(),PAYLOAD)
        self.assertNotEqual((models/'toy/model.bin').stat().st_ino,(backup/'toy/model.bin').stat().st_ino)
        self.assertEqual(list((models/'toy').glob('*.partial')),[])

    def test_rejects_path_traversal_and_escaping_directory_symlink(self):
        models=self.ensure()
        invalid=json.loads(json.dumps(MANIFEST))
        invalid['models']['toy']['files'][0]['path']='../outside.bin'
        with self.assertRaises(ValueError):setup.reuse_weights(models,invalid,['toy'])
        outside=self.root/'outside'
        outside.mkdir()
        (models/'toy').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):setup.reuse_weights(models,MANIFEST,['toy'])
        self.assertFalse((outside/'model.bin').exists())

    def test_cli_forwards_selected_models_verify_only_and_workers(self):
        self.ensure()
        script=self.project/'scripts/common/setup_models.py'
        script.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE,script)
        proc=subprocess.run([sys.executable,str(script),'toy','--verify-only','--workers','3'],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        self.assertEqual(json.loads((self.models/'invoked.json').read_text()),['toy','--workers','3','--verify-only'])

    def test_cli_forwards_company_ca_path_with_spaces(self):
        self.ensure()
        script=self.project/'scripts/common/setup_models.py'
        script.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE,script)
        ca=self.root/'company certificates'/'company-ca.pem'
        proc=subprocess.run([sys.executable,str(script),'toy','--ca-bundle',str(ca)],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        self.assertEqual(json.loads((self.models/'invoked.json').read_text()),['toy','--workers','2','--ca-bundle',str(ca)])

    def test_existing_target_is_not_overwritten_by_reuse(self):
        self.legacy()
        models=self.ensure()
        (models/'toy').mkdir()
        (models/'toy/model.bin').write_bytes(b'existing target stays for downloader verification')
        self.assertEqual(setup.reuse_weights(models,MANIFEST,['toy']),0)
        self.assertEqual((models/'toy/model.bin').read_bytes(),b'existing target stays for downloader verification')


class DispatchTests(unittest.TestCase):
    def load_dispatcher(self):
        spec=importlib.util.spec_from_file_location('tested_download',DOWNLOAD_SOURCE)
        module=importlib.util.module_from_spec(spec)
        fake_yaml=types.SimpleNamespace(safe_load=json.loads)
        with mock.patch.dict(sys.modules,{'yaml':fake_yaml}):spec.loader.exec_module(module)
        return module

    def test_github_dispatch_preserves_keys_status_and_avoids_model_libraries(self):
        module=self.load_dispatcher()
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            module.CONFIG=folder/'download.json'
            module.MODELS_CONFIG=folder/'models.json'
            module.ROOT=folder
            module.CONFIG.write_text(json.dumps({'models':['toy'],'hub':'github'}))
            module.MODELS_CONFIG.write_text(json.dumps({'toy':{'loader':'would_require_torch'}}))
            with mock.patch.object(module.subprocess,'call',return_value=23) as invoke:
                self.assertEqual(module.main(),23)
            self.assertEqual(invoke.call_args.args[0],[sys.executable,str(folder/'scripts/common/setup_models.py'),'toy'])

    def test_empty_selection_does_not_inadvertently_download_all(self):
        module=self.load_dispatcher()
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            module.CONFIG=folder/'download.json'
            module.MODELS_CONFIG=folder/'models.json'
            module.CONFIG.write_text(json.dumps({'models':[],'hub':'github'}))
            module.MODELS_CONFIG.write_text(json.dumps({'toy':{'loader':'would_require_torch'}}))
            with mock.patch.object(module.subprocess,'call') as invoke:
                self.assertEqual(module.main(),0)
                invoke.assert_not_called()


class LegacyDatasetPartsTests(unittest.TestCase):
    def test_release_cache_does_not_interfere_with_dataset_join(self):
        spec=importlib.util.spec_from_file_location('tested_join',SOURCE.with_name('join_parts.py'))
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            data=root/'data';models=root/'models'
            data.mkdir();models.mkdir()
            payload=b'legacy parquet fixture'
            (data/'fixture.parquet.part000').write_bytes(payload)
            (data/'fixture.parquet.sha256').write_text(hashlib.sha256(payload).hexdigest()+'  fixture.parquet\n')
            cache=models/'.downloads/release';cache.mkdir(parents=True)
            (cache/'release--model.safetensors.part000').write_bytes(b'Release part has checksums in manifest.json')
            module.ROOT=root;module.FOLDERS=[data,models]
            # Keep this fixture runnable on Python 3.9; the project's existing join helper targets Python 3.11+.
            module.sha256=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            output=io.StringIO()
            stream=types.SimpleNamespace(reconfigure=lambda **kwargs:None,write=output.write,flush=output.flush)
            with mock.patch.object(module.sys,'stdout',stream):module.main()
            self.assertEqual((data/'fixture.parquet').read_bytes(),payload)
            self.assertFalse((cache/'release--model.safetensors').exists())


if __name__=='__main__':unittest.main(verbosity=2)
