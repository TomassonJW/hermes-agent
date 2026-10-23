"""Managed Linux Python can build sdists without installing its supplier's compiler.

Regression for the missing clang++ build failure in #122402. Explicit compiler
choices and caller-owned interpreters must remain authoritative.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess

import pytest

from pm.environment import managed_environment
import pm._uv
import pm.native_build

pytestmark = pytest.mark.platforms("linux")


@pytest.fixture
def compiler_environment(tmp_path, monkeypatch):
    tools = tmp_path / "host tools with spaces"
    tools.mkdir()
    for name in ("gcc", "g++"):
        executable = tools / name
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    pinned = tmp_path / "managed-python"
    monkeypatch.setattr(pm._uv, "_toolchain", lambda **kwargs: (tmp_path / "uv", pinned))
    config = {"CC": "clang -pthread", "CXX": "clang++ -pthread"}
    monkeypatch.setattr(
        pm.native_build.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, json.dumps(config), ""),
    )
    return tools, pinned, config


def test_managed_python_uses_available_gcc_pair_without_mutating_the_caller(compiler_environment, tmp_path):
    tools, _, _ = compiler_environment
    original = {"PATH": str(tools), "CFLAGS": "-O2", "CXXFLAGS": "-fPIC"}
    before = original.copy()
    built = managed_environment(tmp_path / "candidate", env=original, cache=tmp_path / "cache")
    assert shlex.split(built.env["CC"]) == [str(tools / "gcc"), "-pthread"]
    assert shlex.split(built.env["CXX"]) == [str(tools / "g++"), "-pthread"]
    assert built.env["CFLAGS"] == before["CFLAGS"]
    assert built.env["CXXFLAGS"] == before["CXXFLAGS"]
    assert original == before
    assert not built.destination.exists()


def test_relative_path_compilers_still_exist_after_the_build_changes_directory(compiler_environment, tmp_path, monkeypatch):
    tools, _, _ = compiler_environment
    monkeypatch.chdir(tmp_path)
    built = managed_environment(tmp_path / "candidate", env={"PATH": tools.name}, cache=tmp_path / "cache")
    build_directory = tmp_path / "sdist-build"
    build_directory.mkdir()
    monkeypatch.chdir(build_directory)
    for key, name in (("CC", "gcc"), ("CXX", "g++")):
        command = shlex.split(built.env[key])
        assert Path(command[0]).is_absolute()
        assert command == [str(tools / name), "-pthread"]
        assert Path(command[0]).is_file()


@pytest.mark.parametrize("case", [
    "clang-present", "missing-gxx", "explicit-cc", "explicit-cxx",
    "empty-override", "foreign-python", "different-sysconfig", "empty-path",
    "cross-target-flags", "invalid-sysconfig", "unavailable-probe",
])
def test_other_toolchains_and_explicit_choices_are_not_replaced(compiler_environment, tmp_path, monkeypatch, case):
    tools, pinned, config = compiler_environment
    env = {"PATH": str(tools)}
    build_options = {"python": pinned}

    def clang_present():
        for name in ("clang", "clang++"):
            executable = tools / name
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)

    def unavailable(*args, **kwargs):
        raise OSError("interpreter unavailable")

    setup = {
        "clang-present": clang_present,
        "missing-gxx": lambda: (tools / "g++").unlink(),
        "explicit-cc": lambda: env.update(CC="custom-cc --sysroot=/target"),
        "explicit-cxx": lambda: env.update(CXX="custom-cxx -std=c++20"),
        "empty-override": lambda: env.update(CC=""),
        "foreign-python": lambda: build_options.update(python=tmp_path / "caller-owned-python"),
        "different-sysconfig": lambda: config.update(
            CC="aarch64-linux-gnu-clang -pthread", CXX="aarch64-linux-gnu-clang++ -pthread"),
        "cross-target-flags": lambda: config.update(
            CC="clang --target=aarch64-linux-gnu", CXX="clang++ --target=aarch64-linux-gnu"),
        "invalid-sysconfig": lambda: config.update(CC=None),
        "unavailable-probe": lambda: monkeypatch.setattr(pm.native_build.subprocess, "run", unavailable),
        "empty-path": lambda: env.update(PATH=""),
    }
    setup[case]()
    before = env.copy()
    built = managed_environment(tmp_path / "candidate", env=env, cache=tmp_path / "cache", **build_options)
    assert built.env == before
    assert env == before
