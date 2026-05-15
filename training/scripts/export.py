"""Export quantized TFLite model to EdgeTPU-compiled format and verify quality."""
import shutil
import subprocess
import sys
from pathlib import Path


def parse_compiler_output(stdout: str) -> dict:
    """Parse edgetpu_compiler stdout and return compilation statistics.

    The compiler emits lines like:
      "Total number of operations: 191"
      "CONV_2D    123    Mapped to Edge TPU"
      "RESHAPE    5      More than one subgraph is not supported"

    Returns:
        dict with keys:
          total_ops (int)   — total ops in model
          compiled_ops (int) — ops mapped to EdgeTPU
          cpu_ops (int)     — ops that will run on host CPU
          compiled_pct (float) — percentage mapped to EdgeTPU
    """
    total_ops = 0
    compiled_ops = 0

    for line in stdout.splitlines():
        if "Total number of operations:" in line:
            try:
                total_ops = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
            continue
        if "Mapped to Edge TPU" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    compiled_ops += int(parts[1])
                except ValueError:
                    pass

    cpu_ops = total_ops - compiled_ops
    compiled_pct = (compiled_ops / total_ops * 100.0) if total_ops > 0 else 0.0

    return {
        "total_ops": total_ops,
        "compiled_ops": compiled_ops,
        "cpu_ops": cpu_ops,
        "compiled_pct": compiled_pct,
    }
