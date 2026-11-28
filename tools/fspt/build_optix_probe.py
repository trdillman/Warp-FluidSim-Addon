"""Build the isolated P0 OptiX C ABI probe with locally retained Windows tools.

This is a development feasibility artifact. It is not included in the add-on,
does not install dependencies, and does not launch a GPU workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests" / "acceptance" / "native"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def visual_studio_environment(vcvars):
    # Capture the environment privately. Never write the full inherited
    # environment (which can contain unrelated secrets) to build evidence.
    # Pass the Windows command line as a string. list2cmdline applies C-runtime
    # quote escaping, which is not the quoting language used by cmd.exe /c.
    command = f'cmd.exe /d /s /c ""{vcvars}" x64 >nul && set"'
    output = subprocess.check_output(
        command, text=True, encoding="utf-8", errors="replace", timeout=60
    )
    result = dict(os.environ)
    for line in output.splitlines():
        if "=" in line and not line.startswith("="):
            name, value = line.split("=", 1)
            result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument(
        "--optix-root",
        type=Path,
        default=Path(r"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0"),
    )
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"),
    )
    parser.add_argument(
        "--vcvars",
        type=Path,
        default=Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
            r"\VC\Auxiliary\Build\vcvarsall.bat"
        ),
    )
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This retained-toolchain probe builder qualifies Windows only")
    folder = args.out_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    report_path = folder / "build.json"
    report = {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": digest(__file__),
        "scope": "CPU compilation only; OptiX and Blender execution are NOT_RUN",
        "commands": [],
    }
    try:
        source = args.source_dir.resolve()
        cpp, cuda = source / "fspt_optix_probe.cpp", source / "fspt_optix_probe.cu"
        sdk = args.optix_root.resolve()
        toolkit = args.cuda_root.resolve()
        nvcc = toolkit / "bin" / "nvcc.exe"
        env = visual_studio_environment(args.vcvars.resolve())
        tool_root = Path(env["VCToolsInstallDir"])
        compiler = tool_root / "bin" / "Hostx64" / "x64" / "cl.exe"
        linker = compiler.with_name("link.exe")
        required = [
            cpp,
            cuda,
            source / "OPTIX_PROBE_NOTICES.txt",
            nvcc,
            compiler,
            linker,
            sdk / "include" / "optix.h",
            toolkit / "include" / "cuda.h",
            toolkit / "lib" / "x64" / "cuda.lib",
        ]
        assert all(path.is_file() for path in required), [
            str(p) for p in required if not p.is_file()
        ]
        report["inputs"] = {
            str(path): {"sha256": digest(path), "size": path.stat().st_size}
            for path in required
        }
        report["optix_headers"] = {
            path.relative_to(sdk).as_posix(): digest(path)
            for path in sorted((sdk / "include").rglob("*.h"))
        }
        report["cuda_version"] = json.loads((toolkit / "version.json").read_text())
        report["toolchain"] = {
            name: env.get(name)
            for name in (
                "VCToolsVersion",
                "VCToolsInstallDir",
                "WindowsSdkDir",
                "WindowsSDKVersion",
            )
        }
        ptx = folder / "fspt_optix_probe.ptx"
        library = folder / "fspt_optix_probe.dll"
        commands = [
            [
                str(nvcc),
                "--ptx",
                "--std=c++14",
                "--gpu-architecture=compute_75",
                "-I" + str(sdk / "include"),
                str(cuda),
                "-o",
                str(ptx),
            ],
            [
                str(compiler),
                "/nologo",
                "/LD",
                "/O2",
                "/MD",
                "/EHsc",
                "/std:c++17",
                "/I" + str(sdk / "include"),
                "/I" + str(toolkit / "include"),
                str(cpp),
                "/Fe:" + str(library),
                "/Fo:" + str(folder / "fspt_optix_probe.obj"),
                "/link",
                "/LIBPATH:" + str(toolkit / "lib" / "x64"),
                "cuda.lib",
                "advapi32.lib",
            ],
        ]
        for number, command in enumerate(commands, 1):
            out, err = (
                folder / f"compile-{number}.stdout.log",
                folder / f"compile-{number}.stderr.log",
            )
            with out.open("xb") as stdout, err.open("xb") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=folder,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=120,
                    check=False,
                )
            row = {
                "argv": command,
                "returncode": completed.returncode,
                "stdout": str(out),
                "stderr": str(err),
                "stdout_sha256": digest(out),
                "stderr_sha256": digest(err),
            }
            report["commands"].append(row)
            if completed.returncode:
                raise RuntimeError(
                    f"Compiler {number} exited {completed.returncode}; "
                    f"see {err} and {out}"
                )
        notices = folder / "OPTIX_PROBE_NOTICES.txt"
        notices.write_bytes((source / notices.name).read_bytes())
        report["artifacts"] = {
            key: {
                "path": str(path),
                "sha256": digest(path),
                "size": path.stat().st_size,
            }
            for key, path in (("library", library), ("ptx", ptx), ("notices", notices))
        }
        report["status"] = "VERIFIED"
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FSPT-P0-OPTIX-BUILD-PASS", report_path, flush=True)


if __name__ == "__main__":
    main()
