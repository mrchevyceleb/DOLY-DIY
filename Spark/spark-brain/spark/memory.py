"""Rolling conversation memory, persisted across restarts."""
import collections
import json
import os
import pathlib
import time


class Memory:
    def __init__(self, cfg):
        self.max_turns = cfg["brain"]["history_turns"]
        self.path = pathlib.Path(cfg["state_dir"]) / "memory.jsonl"
        self.history = collections.deque(maxlen=self.max_turns * 2)
        self._lines_written = 0
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").strip().splitlines()
            self._lines_written = len(lines)
            for line in lines[-self.max_turns * 2:]:
                rec = json.loads(line)
                self.history.append({"role": rec["role"], "content": rec["content"]})
        except Exception as e:
            print(f"[memory] load failed: {e}")

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        try:
            self._lines_written += 1
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"role": role, "content": content, "t": time.time()}) + "\n")
            # bounded retention: compact the file once it outgrows the window
            if self._lines_written > self.max_turns * 2 + 64:
                self._compact()
        except Exception as e:
            print(f"[memory] persist failed: {e}")

    def _compact(self):
        """Rewrite the file down to the live window (bounded disk on the Pi)."""
        try:
            lines = self.path.read_text(encoding="utf-8").strip().splitlines()
            keep = lines[-(self.max_turns * 2):]
            self.path.write_text("\n".join(keep) + "\n", encoding="utf-8")
            self._lines_written = len(keep)
        except Exception as e:
            print(f"[memory] compact failed: {e}")

    def messages(self, system_prompt):
        out = [{"role": "system", "content": system_prompt}]
        for turn in self.history:
            if turn["content"]:
                out.append({"role": turn["role"], "content": turn["content"]})
        return out

    def clear(self):
        self.history.clear()
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass
