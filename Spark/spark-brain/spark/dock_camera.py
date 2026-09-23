"""Exclusive camera worker; camera bindings must not stall motor callbacks."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


class DockCamera:
    def __init__(self, stop=None):
        self.stop = stop
        self.process = None
        self.messages = queue.Queue(maxsize=8)

    def __enter__(self):
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "spark.dock_vision", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        def read():
            for line in self.process.stdout:
                if line.startswith("DOCK_OBSERVATION "):
                    self.messages.put(json.loads(line[len("DOCK_OBSERVATION "):]))
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        return self

    def __exit__(self, *exc):
        if self.process:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write("close\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=2)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            self.process.stdin.close()
            self.reader.join(timeout=1)
            self.process.stdout.close()

    def observe(self):
        if self.process.poll() is not None:
            raise RuntimeError("Dock camera stopped")
        self.process.stdin.write("observe\n")
        self.process.stdin.flush()
        until = time.monotonic() + 20
        while time.monotonic() < until:
            if self.stop and self.stop():
                raise InterruptedError("Dock camera cancelled")
            try:
                result = self.messages.get(timeout=.05)
                return SimpleNamespace(**result) if result else None
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError("Dock camera failed")
        raise TimeoutError("Dock camera timed out")
