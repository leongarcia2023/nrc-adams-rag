"""Restore a shared Drive snapshot using its transfer report; no Google login.

pip install gdown; pass the privately supplied FINAL-REPORT.json separately.
The report contains file IDs and hashes. It is not distributed in this repo.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Unsafe archive path')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes destination')
    return path


def verify(path, size, checksum):
    return path.is_file() and not path.is_symlink() and path.stat().st_size == size and digest(path) == checksum


def space(root, amount, reserve):
    if shutil.disk_usage(root).free < amount + reserve * 1e9:
        raise RuntimeError('Not enough free storage for the next restore step')


def emit(state, **values):
    print(json.dumps({'state': state, **values}), flush=True)


def extract_verified(archive_path, root, expected_manifest_hash, reserve):
    with tarfile.open(archive_path, 'r:') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive member')
        manifest_member = archive.getmember('TRANSFER-MANIFEST.json')
        if not manifest_member.isfile() or manifest_member.size > 16 * 1024 * 1024:
            raise ValueError('Invalid archive manifest')
        raw = archive.extractfile(manifest_member).read()
        if hashlib.sha256(raw).hexdigest() != expected_manifest_hash:
            raise ValueError('Archive manifest checksum mismatch')
        manifest = json.loads(raw)
        files = {f['path']: f for f in manifest['files']}
        links = {f['path']: f for f in manifest['symlinks']}
        if set(names) != set(files) | set(links) | {'TRANSFER-MANIFEST.json'}:
            raise ValueError('Archive member inventory mismatch')
        count, total = 0, 0
        for member in members:
            if member.name not in files:
                continue
            expected = files[member.name]
            if not member.isfile() or member.size != expected['size']:
                raise ValueError('Archive member size/type mismatch')
            destination = safe_path(root, member.name)
            if destination.exists() or destination.is_symlink():
                if not verify(destination, expected['size'], expected['sha256']):
                    raise ValueError('Existing destination differs: ' + member.name)
            else:
                space(root, member.size, reserve)
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(destination.name + '.restore-part')
                if partial.is_symlink():
                    raise ValueError('Unexpected restore symlink')
                h = hashlib.sha256()
                with archive.extractfile(member) as source, partial.open('wb') as target:
                    while block := source.read(4 * 1024 * 1024):
                        target.write(block); h.update(block)
                if partial.stat().st_size != expected['size'] or h.hexdigest() != expected['sha256']:
                    raise ValueError('Extracted member checksum mismatch: ' + member.name)
                partial.chmod(0o600)
                os.replace(partial, destination)
            count += 1; total += member.size
            if count % 100 == 0:
                emit('extracting', files=count, bytes=total)
        for name, expected in links.items():
            member = archive.getmember(name)
            if not member.issym() or member.linkname != expected['target']:
                raise ValueError('Archive link mismatch')
            destination = safe_path(root, name)
            target = expected['target']
            if target.startswith('/'):
                # Old snapshot links are rooted in its original outputs directory.
                if '/outputs/' not in target:
                    raise ValueError('Unrecognized absolute snapshot link')
                resolved = safe_path(root, target.split('/outputs/', 1)[1])
            else:
                resolved = (destination.parent / target).resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError('Snapshot link escapes destination')
            link = os.path.relpath(resolved, destination.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                if os.readlink(destination) != link:
                    raise ValueError('Existing symlink differs')
            elif destination.exists():
                raise ValueError('Existing path prevents symlink restoration')
            else:
                destination.symlink_to(link)
        return {'files': count, 'bytes': total, 'symlinks': len(links)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--min-free-gb', type=float, default=5)
    args = parser.parse_args()
    import gdown
    if args.min_free_gb < 1:
        parser.error('Preserve at least 1 GB free')
    root = Path(args.destination).resolve(); root.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.report).read_text())
    info = report['archive']; archive_path = Path(args.archive).resolve()
    marker = root / 'RESTORE-ARCHIVE-VERIFIED.json'
    if not marker.exists():
        if not archive_path.exists():
            space(root, info['size'], args.min_free_gb)
            gdown.download(id=info['archive_file_id'], output=str(archive_path), resume=True, quiet=True)
        emit('verifying_archive', bytes=info['size'])
        if not verify(archive_path, info['size'], info['sha256']):
            raise ValueError('Archive size or checksum mismatch; source retained')
        result = extract_verified(archive_path, root, info['manifest_sha256'], args.min_free_gb)
        marker.write_text(json.dumps({'sha256': info['sha256'], **result}, indent=2))
        # This is the newly downloaded, verified staging copy, not the Drive source.
        archive_path.unlink()
    else:
        if json.loads(marker.read_text())['sha256'] != info['sha256']:
            raise ValueError('Restore receipt belongs to a different snapshot')
    for entry in report['individual_files']:
        path = safe_path(root, entry['path'])
        if path.exists():
            if not verify(path, entry['size'], entry['sha256']):
                raise ValueError('Existing individual file differs')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.download')
        space(root, max(0, entry['size'] - (temporary.stat().st_size if temporary.exists() else 0)), args.min_free_gb)
        emit('downloading_individual', path=entry['path'], bytes=entry['size'])
        gdown.download(id=entry['drive_file_id'], output=str(temporary), resume=True, quiet=True)
        if not verify(temporary, entry['size'], entry['sha256']):
            raise ValueError('Individual file checksum mismatch')
        temporary.chmod(0o600); os.replace(temporary, path)
    summary = {'state': 'complete', 'files': info['verified_files'] + len(report['individual_files']),
               'source_bytes': report['source_bytes_removed'], 'finished_unix': time.time(),
               'free_bytes': shutil.disk_usage(root).free}
    (root / 'RESTORE-COMPLETE.json').write_text(json.dumps(summary, indent=2))
    emit(**summary)


if __name__ == '__main__':
    main()
