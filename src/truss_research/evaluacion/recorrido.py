"""Supervisa el recorrido completo, incluso antes de instalar sus dependencias."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

CHECK_SECONDS = 600
CLEANUP_SECONDS = 5
TIMEOUT_EXIT = 124
KILL_JOB_ON_CLOSE = 0x2000
EXTENDED_JOB_LIMITS = 9
REPLACE_ATTEMPTS = 6
REPLACE_DELAY_SECONDS = 0.02
TREE_FREEZE_PASSES = 3
type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
STAGES = (
    "setup",
    "check",
    "test",
    "prepare",
    "tabpfn-weights",
    "tabpfn-pilot",
    "plan",
    "smoke",
    "study",
    "final",
    "explain",
    "benchmark",
    "report",
    "audit",
    "notebooks",
)


def pipeline_stages(interpolation: bool = False, mode: str = "full") -> tuple[str, ...]:
    stages = tuple(stage for stage in STAGES if mode == "full" or stage != "explain")
    return (*stages, "interpolation") if interpolation else stages


def positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("El valor debe ser un entero positivo")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "check"), default="full")
    parser.add_argument(
        "--accelerator",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Perfil indicado por Makefile; ambos dispositivos usan el entorno raíz",
    )
    parser.add_argument("--estudio", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cpu-workers", type=positive_integer)
    parser.add_argument("--max-seconds", type=positive_integer)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--interpolation-dest", type=Path)
    result = parser.parse_args()
    result.estudio = (
        result.estudio
        or Path(
            "artifacts/comprobacion_flujo"
            if result.mode == "check"
            else "artifacts/comparacion_tres_objetivos_final_001"
        )
    ).resolve()
    result.config = (
        result.config
        or Path("config/smoke.json" if result.mode == "check" else "config/study.json")
    ).resolve()
    if result.mode == "check":
        result.max_seconds = result.max_seconds or CHECK_SECONDS
    return result


def report_root(study: Path) -> Path:
    parent = study.parent
    return (
        (parent.parent if parent.name == "artifacts" else parent)
        / "reports"
        / study.name
    )


def save_record(path: Path, record: dict[str, JsonValue]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            sleep(REPLACE_DELAY_SECONDS * 2**attempt)


@contextmanager
def flow_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run_all.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Otro run-all está usando este estudio") from error
        yield


class ProcessTree:
    """Windows contiene descendientes en un Job Object; POSIX usa una sesión."""

    def __init__(self, command: list[str]) -> None:
        if os.name != "nt" and shutil.which("ps") is None:
            raise RuntimeError("La contención POSIX requiere el comando ps")
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        self.job: tuple[ctypes.CDLL, int] | None = None
        if os.name == "nt":
            try:
                self.attach_windows_job()
            except BaseException:
                self.process.kill()
                self.process.wait(timeout=CLEANUP_SECONDS)
                raise

    def attach_windows_job(self) -> None:
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("min_ws", ctypes.c_size_t),
                ("max_ws", ctypes.c_size_t),
                ("active", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class Limits(ctypes.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        limits = Limits()
        limits.basic.flags = KILL_JOB_ON_CLOSE
        if not handle or not kernel.SetInformationJobObject(
            handle, EXTENDED_JOB_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(
            handle, wintypes.HANDLE(int(self.process._handle))
        ):
            kernel.CloseHandle(handle)
            raise ctypes.WinError(ctypes.get_last_error())
        self.job = (kernel, handle)

    def close(self) -> None:
        deadline = monotonic() + CLEANUP_SECONDS
        if self.job is not None:
            kernel, handle = self.job
            kernel.TerminateJobObject(handle, TIMEOUT_EXIT)
            kernel.CloseHandle(handle)
            self.job = None
        elif os.name != "nt":
            stop_posix_tree(self.process.pid, deadline)
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=max(0.01, deadline - monotonic()))


def posix_descendants(root: int, timeout: float) -> list[int]:
    rows = subprocess.check_output(
        ["ps", "-A", "-o", "pid=", "-o", "ppid="], text=True, timeout=timeout
    )
    children: dict[int, list[int]] = {}
    for line in rows.splitlines():
        pid, parent = map(int, line.split())
        children.setdefault(parent, []).append(pid)
    tree = [root]
    for parent in tree:
        tree.extend(children.get(parent, ()))
    return tree[1:]


def stop_posix_tree(root: int, deadline: float) -> None:
    """Incluye descendientes que cambiaron de sesión, como los kernels Jupyter."""
    captured = [root]
    try:
        try:
            os.kill(root, signal.SIGSTOP)
        except ProcessLookupError:
            pass
        for _ in range(TREE_FREEZE_PASSES):
            found = posix_descendants(root, min(1.0, max(0.01, deadline - monotonic())))
            fresh = [pid for pid in found if pid not in captured]
            if not fresh:
                break
            captured.extend(fresh)
            for pid in fresh:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
    finally:
        for pid in reversed(captured):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.killpg(root, signal.SIGKILL)
        except ProcessLookupError:
            pass


def monitored(command: list[str], context: tuple[float | None, Path]) -> int:
    deadline, output = context
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("Se agotó MAX_SECONDS")
    tree = ProcessTree(command)

    def forward() -> None:
        with output.open("a", encoding="utf-8") as log:
            for line in tree.process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

    reader = threading.Thread(target=forward, daemon=True)
    reader.start()
    try:
        remaining = None if deadline is None else max(0.0, deadline - monotonic())
        try:
            return tree.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "Se agotó MAX_SECONDS; se detuvo el árbol de procesos"
            ) from error
    finally:
        cleanup_started = monotonic()
        tree.close()
        reader.join(timeout=max(0.0, CLEANUP_SECONDS - (monotonic() - cleanup_started)))
        tree.process.stdout.close()


def supervise(args: argparse.Namespace) -> int:
    start = monotonic()
    interpolation = getattr(args, "interpolation", False)
    interpolation_dest = getattr(
        args, "interpolation_dest", None
    ) or args.estudio.with_name(args.estudio.name + "_interpolacion")
    if interpolation and args.mode != "full":
        raise ValueError("La interpolación de run-all requiere RUN_MODE=full")
    deadline = None if args.max_seconds is None else start + args.max_seconds
    if args.plan:
        receipt = report_root(args.estudio) / "run_all.json"
        cached = (
            json.loads(receipt.read_text(encoding="utf-8")).get("stages", {})
            if receipt.is_file()
            else {}
        )
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "study": str(args.estudio),
                    "config": str(args.config),
                    "max_seconds": args.max_seconds,
                    "stages": pipeline_stages(interpolation, args.mode),
                    "interpolation_destination": str(interpolation_dest)
                    if interpolation
                    else None,
                    "reuse": {
                        name: "always"
                        if name in ("setup", "check")
                        else "verify_saved_inputs_and_outputs"
                        if cached.get(name, {}).get("status") == "complete"
                        else "required"
                        for name in pipeline_stages(interpolation, args.mode)
                    },
                },
                indent=2,
            )
        )
        return 0
    root = report_root(args.estudio)
    with flow_lock(root):
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        number = (
            max(
                (
                    int(p.stem.removeprefix("run_all_"))
                    for p in logs.glob("run_all_*.json")
                    if p.stem.removeprefix("run_all_").isdigit()
                ),
                default=0,
            )
            + 1
        )
        path = logs / f"run_all_{number:03d}.json"
        record = {
            "status": "running",
            "mode": args.mode,
            "study": str(args.estudio),
            "max_seconds": args.max_seconds,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "stage": "setup",
        }
        save_record(path, record)
        code = 1
        try:
            setup_started = monotonic()
            code = monitored(
                ["uv", "sync", "--locked"],
                (deadline, path.with_suffix(".log")),
            )
            if code != 0:
                raise RuntimeError(f"setup terminó con código {code}")
            record["setup_seconds"] = monotonic() - setup_started
            global_path = root / "run_all.json"
            global_record = (
                json.loads(global_path.read_text(encoding="utf-8"))
                if global_path.is_file()
                else {"stages": {}}
            )
            global_record["stages"]["setup"] = {
                "status": "complete",
                "wall_seconds": record["setup_seconds"],
                "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
            }
            save_record(global_path, global_record)
            record["stage"] = "pipeline"
            save_record(path, record)
            command = [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "truss_research.evaluacion.campana",
                "--mode",
                args.mode,
                "--estudio",
                str(args.estudio),
                "--config",
                str(args.config),
                "--accelerator",
                args.accelerator,
            ]
            if args.cpu_workers is not None:
                command.extend(["--cpu-workers", str(args.cpu_workers)])
            if interpolation:
                command.extend(
                    ["--interpolation", "--interpolation-dest", str(interpolation_dest)]
                )
            code = monitored(command, (deadline, path.with_suffix(".log")))
            record["status"] = "complete" if code == 0 else "failed"
        except TimeoutError as error:
            record.update(status="timed_out", error=str(error))
            code = TIMEOUT_EXIT
        except KeyboardInterrupt:
            record.update(status="interrupted", error="Interrumpido por el usuario")
            code = 130
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            code = code or 1
        finally:
            record.update(wall_seconds=monotonic() - start, exit_code=code)
            save_record(path, record)
            if code != 0 and (root / "run_all.json").is_file():
                saved = json.loads((root / "run_all.json").read_text(encoding="utf-8"))
                saved["status"] = record["status"]
                saved["technical_complete"] = False
                current = saved.get("current_stage")
                if (
                    current
                    and saved["stages"].get(current, {}).get("status") == "running"
                ):
                    saved["stages"][current].update(
                        status=record["status"],
                        error=record.get("error", "El proceso terminó con error"),
                    )
                save_record(root / "run_all.json", saved)
            print(
                f"run-all: {record['status']}; {record['wall_seconds']:.1f} s; {path}",
                flush=True,
            )
        return code


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    try:
        raise SystemExit(supervise(arguments()))
    except (ValueError, RuntimeError) as failure:
        print(str(failure), file=sys.stderr)
        raise SystemExit(1) from failure
