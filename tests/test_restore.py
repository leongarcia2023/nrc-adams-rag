import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('restore', Path(__file__).resolve().parents[1] / 'scripts/restore_drive_archive.py')
restore = importlib.util.module_from_spec(spec); spec.loader.exec_module(restore)


class RestoreTests(unittest.TestCase):
    def test_roundtrip_checks_files_and_relocates_snapshot_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive_path = root / 'snapshot.tar'; destination = root / 'restore'; destination.mkdir()
            name = 'adams-rag/data/document.md'; content = b'# Test\nSource content\n'
            manifest = {'files': [{'path': name, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}],
                'symlinks': [{'path': 'adams-rag/data/link.md', 'target': '/old/outputs/' + name}]}
            raw = json.dumps(manifest).encode()
            with tarfile.open(archive_path, 'w') as archive:
                for path, data in [(name, content), ('TRANSFER-MANIFEST.json', raw)]:
                    info = tarfile.TarInfo(path); info.size = len(data); archive.addfile(info, io.BytesIO(data))
                info = tarfile.TarInfo('adams-rag/data/link.md'); info.type = tarfile.SYMTYPE
                info.linkname = manifest['symlinks'][0]['target']; archive.addfile(info)
            checksum = hashlib.sha256(raw).hexdigest()
            self.assertEqual(restore.extract_verified(archive_path, destination, checksum, 0)['files'], 1)
            self.assertEqual((destination / name).read_bytes(), content)
            self.assertEqual((destination / 'adams-rag/data/link.md').read_bytes(), content)
            restore.extract_verified(archive_path, destination, checksum, 0)
            (destination / name).write_bytes(b'changed')
            with self.assertRaises(ValueError): restore.extract_verified(archive_path, destination, checksum, 0)

    def test_paths_and_manifest_are_checked_before_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['/etc/passwd', '../escape', 'a/../../escape']:
                with self.assertRaises(ValueError): restore.safe_path(root, name)
            (root / 'outside').symlink_to(root.parent)
            with self.assertRaises(ValueError): restore.safe_path(root, 'outside/escape')
