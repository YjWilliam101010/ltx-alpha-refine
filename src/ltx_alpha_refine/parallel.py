# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
"""Frame-local CPU processing with shared, serialized GPU inference. No model/threshold change."""
import copy, threading, torch, time

class FrameWorkers:

    def __init__(self, prototype):
        self.prototype = prototype
        self.local = threading.local()
        self.instances = []
        self.guard = threading.Lock()
        self.gpu_lock = threading.Lock()
        owner = self

        class Model:

            def __call__(self, *args, **kwargs):
                with owner.gpu_lock:
                    result = owner.prototype.model(*args, **kwargs)
                    torch.cuda.current_stream().synchronize()
                    return result
        self.shared = Model()

    def repair(self, *args, **kwargs):
        if not hasattr(self.local, 'instance'):
            obj = copy.copy(self.prototype)
            obj.model = self.shared
            obj.secondary_forwards = 0
            obj.secondary_seconds = 0.0
            obj.secondary_commits = 0
            self.local.instance = obj
            with self.guard:
                self.instances.append(obj)
        return self.local.instance.repair(*args, **kwargs)

    def totals(self):
        return {name: sum((getattr(obj, name) for obj in self.instances)) for name in ['secondary_forwards', 'secondary_seconds', 'secondary_commits']}
