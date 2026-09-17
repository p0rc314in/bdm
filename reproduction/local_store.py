"""Filesystem result/checkpoint transport for the extracted experiment runners.

No hosted account is needed. Artifact publication copies bytes before updating
an atomic catalog; recovery reads verify those bytes. One writer owns a run.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import uuid

run = None


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


class Record(dict):
    def __init__(self, path, value=None):
        self.path = path
        super().__init__(value or {})
    def update(self, value=None, **kwargs):
        kwargs.pop('allow_val_change', None)
        super().update(value or {}, **kwargs)
        atomic(self.path, self)
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        atomic(self.path, self)


class Artifact:
    def __init__(self, name, type, metadata=None):
        self.name, self.type, self.metadata = name, type, metadata or {}
        self.ttl, self.files, self.aliases = None, {}, []
    def add_file(self, path, name=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        name = name or path.name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('artifact member must be a relative path')
        self.files[name] = path
    def add_dir(self, path, name=None):
        path = Path(path)
        for item in sorted(path.rglob('*')):
            if item.is_file() and '__pycache__' not in item.parts:
                self.add_file(item, str(Path(name or '') / item.relative_to(path)))


class Saved:
    def __init__(self, root, row):
        self.root, self.row = Path(root), row
        self.__dict__.update(row)
        self.qualified_name = self.name
        self.state = 'COMMITTED'
        self.ttl = None
    def wait(self, **kwargs):
        return self
    def save(self):
        pass  # Retention is local; permanent files are never silently expired.
    def download(self, root=None):
        source = self.root / self.row['directory']
        for name, expected in self.row['files'].items():
            if digest(source / name) != expected:
                raise ValueError(f'artifact checksum mismatch: {self.name}/{name}')
        if root is not None:
            shutil.copytree(source, root, dirs_exist_ok=True)
            return str(root)
        return str(source)


class Inputs:
    def __init__(self, path):
        self.path = Path(path)
    def download(self, **kwargs):
        if not (self.path / 'manifest.json').is_file() and not (self.path / 'training/manifest.json').is_file():
            raise FileNotFoundError(f'Prepare inputs first: {self.path}')
        return str(self.path)


class Run:
    def __init__(self, config=None, **kwargs):
        self.root = Path(os.environ['BDM_RUN_DIR']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.id = self.root.name
        self.entity, self.project, self.path = 'local', 'reproduction', self.id
        self.catalog_path = self.root / 'artifacts.json'
        self.catalog = json.loads(self.catalog_path.read_text()) if self.catalog_path.exists() else []
        self.resumed = bool(self.catalog)
        self.config = Record(self.root / 'config.json', config)
        atomic(self.root / 'config.json', self.config)
        self.summary = Record(self.root / 'summary.json')
    def __enter__(self):
        global run
        run = self
        return self
    def __exit__(self, typ, value, traceback):
        global run
        self.summary.update(status='failed' if typ else 'finished')
        run = None
    def define_metric(self, *args, **kwargs):
        pass
    def log(self, values, **kwargs):
        with (self.root / 'metrics.jsonl').open('a') as output:
            output.write(json.dumps(values, default=str) + '\n')
    def use_artifact(self, reference, **kwargs):
        registry = json.loads(os.environ['BDM_INPUTS'])
        if reference not in registry:
            raise ValueError(f'Unregistered input: {reference}')
        return Inputs(registry[reference])
    def logged_artifacts(self):
        return [Saved(self.root, row) for row in self.catalog]
    def log_artifact(self, artifact, aliases=None):
        versions = [r for r in self.catalog if r['name'].split(':')[0] == artifact.name]
        version = max((r['version'] for r in versions), default=-1) + 1
        name = artifact.name + ':v' + str(version)
        directory = Path('artifacts') / (artifact.name + '-v' + str(version) + '-' + uuid.uuid4().hex)
        temporary = self.root / ('publish-' + uuid.uuid4().hex)
        temporary.mkdir()
        checksums = {}
        for member, path in artifact.files.items():
            target = temporary / member
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            checksums[member] = digest(target)
        final = self.root / directory
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(final)
        aliases = aliases or []
        for row in versions:
            row['aliases'] = [a for a in row['aliases'] if a not in aliases]
        row = dict(id=name, name=name, version=version, type=artifact.type,
                   metadata=artifact.metadata, aliases=aliases,
                   directory=str(directory), files=checksums)
        self.catalog.append(row)
        atomic(self.catalog_path, self.catalog)
        return Saved(self.root, row)


def init(**kwargs):
    return Run(**kwargs)


class Api:
    def run(self, path):
        if run is None:
            raise RuntimeError('no active local run')
        return run
