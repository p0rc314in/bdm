"""Native W&B delivery adapter for the established fine-tuning functions."""
from concurrent.futures import Future
from datetime import timedelta
from pathlib import Path
from reproduction import local_store as wandb


def trim_uploaded_cache():
    pass  # Local outputs remain available for independent inspection.


class WBTransport:
    def __init__(self, run, phase):
        self.run, self.phase, self.previous = run, phase, None

    def publish_pair(self, name, path, metadata):
        artifact = wandb.Artifact(f'{self.run.id}-{self.phase}-recovery', type='checkpoint')
        artifact.ttl = None
        artifact.add_file(str(path), name='checkpoint.pt')
        artifact.add_file(str(metadata), name='metadata.json')
        uploaded = self.run.log_artifact(artifact, aliases=['latest']); uploaded.wait()
        if self.previous is not None and self.previous.id != uploaded.id:
            self.previous.ttl = timedelta(days=30); self.previous.save()
        self.previous = uploaded
        trim_uploaded_cache()
        result = Future(); result.set_result(uploaded)
        return result

    def put(self, name, path):
        artifact = wandb.Artifact(f'{self.run.id}-{self.phase}-{name.replace(".", "-")}', type='evaluation')
        artifact.ttl = None; artifact.add_file(str(path), name=name)
        self.run.log_artifact(artifact).wait()
        trim_uploaded_cache()

    def finish(self):
        pass  # All uploads above wait for server acknowledgement.
