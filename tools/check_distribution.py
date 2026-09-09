"""Check TensorDev release archives for completeness, purity, and build debris."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from packaging.utils import parse_sdist_filename, parse_wheel_filename


_FORBIDDEN_PARTS = {
    ".git", ".DS_Store", ".pytest_cache", "__pycache__",
    ".ipynb_checkpoints", "notebooks", "academia",
}
_CACHE_SUFFIXES = {".pyc", ".pyo", ".nbc", ".nbi", ".ipynb"}
_LIBRARY_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_paths(names):
    _require(len(names) == len(set(names)), "duplicate archive members")
    for name in names:
        path = PurePosixPath(name)
        _require(
            not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            f"unsafe archive path: {name}",
        )
        _require(
            not _FORBIDDEN_PARTS.intersection(path.parts)
            and path.suffix not in _CACHE_SUFFIXES,
            f"unexpected build or notebook artifact: {name}",
        )


def check_distribution(path: Path, *, source_root: Path) -> None:
    modules = {
        file.relative_to(source_root / "src").as_posix()
        for file in (source_root / "src" / "tensordev").rglob("*.py")
        if "__pycache__" not in file.parts
    }
    _require("tensordev/__init__.py" in modules, "missing source package")
    if path.suffix == ".whl":
        _check_wheel(path, modules)
    else:
        _check_sdist(path, modules)


def _check_wheel(path, modules):
    name, version, _, tags = parse_wheel_filename(path.name)
    _require(name == "tensordev", f"unexpected distribution: {name}")
    _require(
        all(tag.interpreter == "py3" and tag.abi == "none" for tag in tags),
        "TensorDev wheels must use the py3-none ABI tag",
    )
    pure = all(tag.platform == "any" for tag in tags)
    info = f"tensordev-{version}.dist-info"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _check_paths(names)
        files = {name for name in names if not name.endswith("/")}
        required = modules | {
            f"{info}/WHEEL", f"{info}/METADATA", f"{info}/RECORD",
            f"{info}/licenses/LICENSE",
        }
        _require(required <= files, f"missing wheel files: {sorted(required - files)}")
        for name in files:
            _require(
                PurePosixPath(name).parts[0] in {"tensordev", "tensordev.libs", info},
                f"unexpected wheel package: {name}",
            )
        metadata = BytesParser().parsebytes(archive.read(f"{info}/METADATA"))
        wheel = BytesParser().parsebytes(archive.read(f"{info}/WHEEL"))
        _require(metadata["Name"] == "tensordev", "incorrect METADATA Name")
        _require(metadata["Version"] == str(version), "incorrect METADATA Version")
        _require(
            wheel["Root-Is-Purelib"] == str(pure).lower(),
            "wheel purity does not match its platform tag",
        )
        _require(
            set(wheel.get_all("Tag", [])) == {str(tag) for tag in tags},
            "WHEEL tags do not match the filename",
        )
        native = {name for name in files if name.startswith("tensordev/_native_cpu/")}
        libraries = {
            name for name in files
            if PurePosixPath(name).suffix in _LIBRARY_SUFFIXES
            or ".so." in PurePosixPath(name).name
        }
        if pure:
            _require(
                not native and not libraries
                and not any(name.startswith("tensordev.libs/") for name in files),
                "pure wheel contains native files",
            )
        else:
            handlers = native & libraries
            _require(len(handlers) == 1, "native wheel must contain one FFI library")
            _require(
                PurePosixPath(next(iter(handlers))).name
                in {"_tensordev_native_cpu.so", "_tensordev_native_cpu.dylib"},
                "unexpected FFI library name",
            )
            _require(
                native == handlers | {
                    "tensordev/_native_cpu/__init__.py",
                    "tensordev/_native_cpu/_loader.py",
                },
                "native wheel has missing or unexpected loader files",
            )


def _check_sdist(path, modules):
    name, version = parse_sdist_filename(path.name)
    _require(name == "tensordev", f"unexpected distribution: {name}")
    prefix = f"tensordev-{version}/"
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        _check_paths([member.name for member in members])
        _require(all(member.isfile() for member in members), "sdist contains non-files")
        _require(
            all(member.name.startswith(prefix) for member in members),
            "sdist has an incorrect root directory",
        )
        files = {member.name[len(prefix):] for member in members}
        required = {f"src/{module}" for module in modules} | {
            "pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE", "PKG-INFO",
            "native/pyproject.toml", "native/CMakeLists.txt", "native/cpp/kernels.cc",
            "native/src/tensordev_native_cpu/__init__.py",
            "native/src/tensordev_native_cpu/_loader.py",
        }
        _require(required <= files, f"missing sdist files: {sorted(required - files)}")
        for name in files:
            _require(
                name in required | {"native/README.md", "native/LICENSE"}
                or (name.startswith(("tests/", "native/tests/")) and name.endswith(".py"))
                or name == "tools/check_distribution.py",
                f"unexpected sdist file: {name}",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args()
    for archive in args.archives:
        check_distribution(archive, source_root=Path(__file__).resolve().parents[1])
        print(f"Verified {archive.name}")


if __name__ == "__main__":
    main()
