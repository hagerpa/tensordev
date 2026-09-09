from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "check_distribution", Path(__file__).resolve().parents[2] / "tools/check_distribution.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKER)

_MODULES = ("tensordev/__init__.py", "tensordev/core/__init__.py")
_DIST_INFO = "tensordev-0.1.0.dist-info"
_NATIVE_LIBRARY = "tensordev/_native_cpu/_tensordev_native_cpu.dylib"
_METADATA = b"Metadata-Version: 2.4\nName: tensordev\nVersion: 0.1.0\n"


@pytest.fixture
def source_root(tmp_path):
    source = tmp_path / "source"
    for module in _MODULES:
        path = source / "src" / module
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return source


def _wheel_entries(*, native):
    tag = "py3-none-macosx_11_0_arm64" if native else "py3-none-any"
    entries = dict.fromkeys(_MODULES, b"")
    entries.update(
        {
            f"{_DIST_INFO}/WHEEL": (
                "Wheel-Version: 1.0\n"
                f"Root-Is-Purelib: {str(not native).lower()}\nTag: {tag}\n"
            ).encode(),
            f"{_DIST_INFO}/METADATA": _METADATA,
            f"{_DIST_INFO}/RECORD": b"",
            f"{_DIST_INFO}/licenses/LICENSE": b"Apache-2.0\n",
        }
    )
    if native:
        entries.update(
            {
                "tensordev/_native_cpu/__init__.py": b"",
                "tensordev/_native_cpu/_loader.py": b"",
                _NATIVE_LIBRARY: b"library",
            }
        )
    return tag, entries


def _write_wheel(tmp_path, tag, entries):
    path = tmp_path / f"tensordev-0.1.0-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize("native", [False, True], ids=["pure", "native"])
def test_valid_wheel(tmp_path, source_root, native):
    tag, entries = _wheel_entries(native=native)
    _CHECKER.check_distribution(
        _write_wheel(tmp_path, tag, entries), source_root=source_root
    )


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
@pytest.mark.parametrize("multiple_tags", [False, True], ids=["single-tag", "multi-tag"])
def test_repaired_linux_wheel(tmp_path, source_root, architecture, multiple_tags):
    original_tag, entries = _wheel_entries(native=True)
    platforms = [f"manylinux_2_28_{architecture}"]
    if multiple_tags:
        platforms.append(f"manylinux_2_29_{architecture}")
    tag = f"py3-none-{'.'.join(platforms)}"
    entries[f"{_DIST_INFO}/WHEEL"] = entries[f"{_DIST_INFO}/WHEEL"].replace(
        f"Tag: {original_tag}\n".encode(),
        "".join(f"Tag: py3-none-{platform}\n" for platform in platforms).encode(),
    )
    entries["tensordev/_native_cpu/_tensordev_native_cpu.so"] = entries.pop(
        _NATIVE_LIBRARY
    )
    entries["tensordev.libs/libstdc++-01234567.so.6"] = b"dependency"
    _CHECKER.check_distribution(
        _write_wheel(tmp_path, tag, entries), source_root=source_root
    )


@pytest.mark.parametrize(
    "native, removed, added",
    [
        (False, _MODULES[1], None),
        (False, None, "notebooks/play.ipynb"),
        (False, None, "tensordev/.DS_Store"),
        (False, None, "tensordev/__pycache__/__init__.cpython-312.pyc"),
        (False, None, "tensordev.libs/libstdc++-01234567.so.6"),
        (True, _NATIVE_LIBRARY, None),
        (True, None, "tensordev_native_cpu/__init__.py"),
    ],
    ids=[
        "missing-module", "notebook", "finder-file", "cache",
        "native-dependency", "missing-library", "companion",
    ],
)
def test_invalid_wheel_contents(tmp_path, source_root, native, removed, added):
    tag, entries = _wheel_entries(native=native)
    if removed:
        del entries[removed]
    if added:
        entries[added] = b""
    with pytest.raises(ValueError):
        _CHECKER.check_distribution(
            _write_wheel(tmp_path, tag, entries), source_root=source_root
        )


@pytest.mark.parametrize("native", [False, True], ids=["pure", "native"])
@pytest.mark.parametrize("field", ["purity", "tag"])
def test_invalid_wheel_metadata(tmp_path, source_root, native, field):
    tag, entries = _wheel_entries(native=native)
    wheel_key = f"{_DIST_INFO}/WHEEL"
    if field == "purity":
        entries[wheel_key] = entries[wheel_key].replace(
            f"Root-Is-Purelib: {str(not native).lower()}".encode(),
            f"Root-Is-Purelib: {str(native).lower()}".encode(),
        )
    else:
        entries[wheel_key] = entries[wheel_key].replace(
            f"Tag: {tag}".encode(), b"Tag: cp312-cp312-win_amd64"
        )
    with pytest.raises(ValueError):
        _CHECKER.check_distribution(
            _write_wheel(tmp_path, tag, entries), source_root=source_root
        )


def _sdist_entries():
    entries = dict.fromkeys(
        [
            "pyproject.toml",
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
            "native/CMakeLists.txt",
            "native/cpp/kernels.cc",
            "native/src/tensordev_native_cpu/__init__.py",
            "native/src/tensordev_native_cpu/_loader.py",
            "native/pyproject.toml",
            *(f"src/{module}" for module in _MODULES),
        ],
        b"",
    )
    entries["PKG-INFO"] = _METADATA
    return entries


def _write_sdist(tmp_path, entries):
    path = tmp_path / "tensordev-0.1.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries.items():
            member = tarfile.TarInfo(f"tensordev-0.1.0/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_valid_sdist(tmp_path, source_root):
    _CHECKER.check_distribution(
        _write_sdist(tmp_path, _sdist_entries()), source_root=source_root
    )


@pytest.mark.parametrize(
    "removed, added",
    [
        (f"src/{_MODULES[1]}", None),
        ("native/cpp/kernels.cc", None),
        (None, "tests/example.ipynb"),
        (None, "native/.pytest_cache/README.md"),
        (None, "native/src/tensordev_native_cpu/_tensordev_native_cpu.so"),
        (None, "academia/main.tex"),
        (None, "src/tensordev/.DS_Store"),
    ],
    ids=[
        "missing-module", "missing-kernel", "notebook", "cache",
        "library", "academia", "finder-file",
    ],
)
def test_invalid_sdist_contents(tmp_path, source_root, removed, added):
    entries = _sdist_entries()
    if removed:
        del entries[removed]
    if added:
        entries[added] = b""
    with pytest.raises(ValueError):
        _CHECKER.check_distribution(
            _write_sdist(tmp_path, entries), source_root=source_root
        )
