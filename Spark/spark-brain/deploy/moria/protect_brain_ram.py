"""Prevent further swap in LM Studio's dedicated desktop scope, without restart.

Run as the desktop owner after launching LM Studio. Refuse a shared scope.
"""
import json
import os
from pathlib import Path
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parent(pid):
    for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
        if line.startswith('PPid:'):
            return int(line.split()[1])
    raise RuntimeError('Process disappeared')


def protect():
    groups = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            exe = str((proc / 'exe').readlink())
            if '/.lmstudio/extensions/backends/' not in exe or not exe.endswith('/llama-server'):
                continue
            group = (proc / 'cgroup').read_text().strip().removeprefix('0::')
            require(group.startswith(f'/user.slice/user-{os.getuid()}.slice/'), 'Not the desktop owner')
            unit = group.rsplit('/', 1)[-1]
            require(unit.startswith('app-') and unit.endswith('.scope'), 'Not a dedicated app scope')
            groups[group] = int(proc.name)
        except (FileNotFoundError, PermissionError):
            continue
    for group, model_pid in groups.items():
        scope = Path('/sys/fs/cgroup') / group.lstrip('/')
        # The swap limit covers descendants too; validate the whole subtree.
        pids = {int(pid) for file in [scope / 'cgroup.procs', *scope.rglob('cgroup.procs')]
                for pid in file.read_text().split()}
        root = model_pid
        while parent(root) in pids:
            root = parent(root)
        require('lm-studio' in (Path('/proc') / str(root) / 'comm').read_text().lower(), 'Shared launcher scope')
        for pid in pids:
            ancestor = pid
            seen = set()
            while ancestor != root and ancestor in pids and ancestor not in seen:
                seen.add(ancestor)
                ancestor = parent(ancestor)
            require(ancestor == root, 'Scope includes an unrelated process')
        unit = group.rsplit('/', 1)[-1]
        before = int((scope / 'memory.swap.current').read_text())
        subprocess.run(['systemctl', '--user', 'set-property', '--runtime', unit, 'MemorySwapMax=0'], check=True)
        require((scope / 'memory.swap.max').read_text().strip() == '0', 'Swap limit was not applied')
        print(json.dumps({'protected_scope': unit, 'existing_swap_mb': round(before / 1048576),
                          'model_pid': model_pid, 'scope_processes': len(pids)}), flush=True)
    if not groups:
        print('No resident LM Studio worker; apply again after loading the model.')


if __name__ == '__main__':
    protect()
