"""Ejecuta pytest en un proceso con recursos numéricos controlados."""

import os
import subprocess
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def main() -> int:
    environment = {**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1")}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *sys.argv[1:]], env=environment, check=False
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
