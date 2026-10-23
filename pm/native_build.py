"""Native compiler environment for dependency builds from a source checkout.

Windows ARM64 has no wheel for parts of the locked closure (cryptography), so
every sync there compiles from sdists and needs MSVC, Clang, Rust and static
OpenSSL. PM owns the sync, so PM prepares that environment. Otherwise only
callers that remembered to (source activation) could build, and
install.ps1, `hermes update` and repair failed in openssl-sys.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from pm.progress import run_contained

_PROVIDER = Path("scripts/build/windows-deps.ps1")


def pinned_compiler_overrides(python: Path, env: Mapping[str, str]) -> dict[str, str]:
    """Use an available host GCC pair when managed Linux Python names missing Clang.

    Python-build-standalone records its supplier's compiler in sysconfig; that
    compiler need not be installed on the user's machine. Do not replace an
    explicit choice, a cross compiler, or the defaults of a caller-owned Python.
    The caller establishes that ``python`` is PM's pinned interpreter.
    """
    from hermes_platform.host.facts import os_family
    from hermes_platform.resolver import LookupContext, locate_command

    if os_family() != "linux" or "CC" in env or "CXX" in env:
        return {}
    context = LookupContext(path=env.get("PATH", ""))
    if all(locate_command(name, context).found for name in ("clang", "clang++")):
        return {}
    replacements = {key: locate_command(name, context).command
                    for key, name in (("CC", "gcc"), ("CXX", "g++"))}
    if not all(replacements.values()):
        return {}
    try:
        result = subprocess.run(
            [str(python), "-I", "-S", "-c",
             "import json,sysconfig;print(json.dumps({k:sysconfig.get_config_var(k) for k in ('CC','CXX')}))"],
            env=dict(env), capture_output=True, text=True, check=True, timeout=5,
        )
        config = json.loads(result.stdout)
        if not isinstance(config, dict) or not all(isinstance(config.get(key), str) for key in replacements):
            return {}
        commands = {key: shlex.split(config[key]) for key in replacements}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    if commands["CC"][:1] != ["clang"] or commands["CXX"][:1] != ["clang++"]:
        return {}
    # Only the supplier's ordinary host defaults are interchangeable. Target,
    # sysroot and other custom toolchain flags must never be reinterpreted.
    if any(command[1:] not in ([], ["-pthread"]) for command in commands.values()):
        return {}
    # uv changes cwd for sdist builds. Anchor PATH hits without resolving
    # symlinks, whose driver basename can be meaningful (for example ccache).
    return {key: shlex.join([str(Path(replacements[key][0]).absolute()), *commands[key][1:]])
            for key in replacements}


def prepare_windows_environment(*, source: Path, state: Path, env: Mapping[str, str]) -> dict[str, str]:
    """The distribution adapter decides whether this target needs ARM64 tools."""
    shell = shutil.which("powershell", path=env.get("PATH")) or shutil.which("pwsh", path=env.get("PATH"))
    if shell is None:
        raise FileNotFoundError("PowerShell is required to prepare Windows ARM64 build dependencies")
    state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="environment-", dir=state) as scratch:
        output = Path(scratch) / "environment.json"
        # A cold vcpkg clone and OpenSSL build print thousands of lines; the
        # user needs the step and its failure, not the patch log.
        run_contained(
            [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
             str(source / _PROVIDER), "-StateRoot", str(state),
             "-EnvironmentFile", str(output)],
            "Preparing Windows ARM64 build tools", indent="  ",
            cwd=source, env=dict(env), stdin=subprocess.DEVNULL,
        )
        prepared = json.loads(output.read_text(encoding="utf-8-sig"))
    if not isinstance(prepared, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in prepared.items()):
        raise ValueError("Windows build dependency provider returned an invalid environment")
    return prepared


def source_build_environment(source: Path) -> dict[str, str] | None:
    """The environment a dependency build of ``source`` needs on this host.

    None means the ambient environment suffices. Only a checkout carries the
    provider; a payload's dependencies are prebuilt, so it needs no compiler.
    The state root is the store's parent, the one source setup has always
    used, so an existing vcpkg/OpenSSL build is reused rather than repeated.
    """
    from pm.paths import store_root
    from pm.store import current_target

    if current_target() != "win32-arm64" or not (source / _PROVIDER).is_file():
        return None
    from pm.index_config import bridged_index_settings

    prepared = prepare_windows_environment(source=source, state=store_root().parent, env=os.environ)
    # managed_environment translates pip's index knobs only for the ambient
    # environment; this one replaces it, so mirrors must ride along.
    prepared.update(bridged_index_settings(os.environ))
    return prepared
