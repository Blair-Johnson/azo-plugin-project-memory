"""Opt-in diagnostic fault: unavailable shared memory store, not real NFS."""
import os
from pathlib import Path

flag = os.environ.get("AZO_MEMORY_TEST_OUTAGE")
if flag:
    from tmux_pilot.fs_store import RecordStore

    original_init = RecordStore.__init__

    def guarded_init(self, root, *args, **kwargs):
        path = Path(root)
        # The diagnostic supplies this unique project. Never affect other stores.
        project = os.environ.get("AZO_MEMORY_TEST_PROJECT", "")
        if project and project in path.parts and "plugin-data" in path.parts and Path(flag).exists():
            raise OSError("diagnostic: shared project-memory storage is unavailable")
        return original_init(self, root, *args, **kwargs)

    RecordStore.__init__ = guarded_init
