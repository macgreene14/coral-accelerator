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


_MIN_EDGETPU_PCT = 95.0  # warn if fewer than this % of ops compile to EdgeTPU


def compile_edgetpu(quant_path: Path, output_dir: Path) -> Path:
    """Compile quantized TFLite model for EdgeTPU.

    Calls edgetpu_compiler, parses output, asserts ≥95% ops compiled.
    Falls back to copying the quantized model if compiler is not found.

    Returns:
        Path to pv_detector_edgetpu.tflite in output_dir.
    """
    compiler = shutil.which("edgetpu_compiler")
    fallback_path = output_dir / "pv_detector_edgetpu.tflite"

    if not compiler:
        print("WARNING: edgetpu_compiler not found in PATH.", file=sys.stderr)
        print("Falling back to quantized model (no EdgeTPU acceleration).", file=sys.stderr)
        shutil.copy2(quant_path, fallback_path)
        return fallback_path

    print(f"Compiling for EdgeTPU: {quant_path.name}")
    result = subprocess.run(
        [compiler, str(quant_path), "-o", str(output_dir)],
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    stats = parse_compiler_output(combined)

    print(
        f"EdgeTPU: {stats['compiled_ops']}/{stats['total_ops']} ops "
        f"({stats['compiled_pct']:.1f}%) mapped to EdgeTPU"
    )

    if stats["total_ops"] > 0 and stats["compiled_pct"] < _MIN_EDGETPU_PCT:
        print(
            f"WARNING: Only {stats['compiled_pct']:.1f}% of ops run on EdgeTPU "
            f"({stats['cpu_ops']} ops fall back to CPU).",
            file=sys.stderr,
        )
        print(
            "EfficientDet-Lite0 should achieve ~100%. "
            "Check for unsupported ops in the compiler log.",
            file=sys.stderr,
        )

    # edgetpu_compiler writes <stem>_edgetpu.tflite; rename to canonical name
    compiled_src = output_dir / (quant_path.stem + "_edgetpu.tflite")
    if compiled_src.exists() and compiled_src != fallback_path:
        compiled_src.rename(fallback_path)
    elif not fallback_path.exists():
        # compiler may have failed silently — fall back
        print("WARNING: compiler produced no output. Using quantized model.", file=sys.stderr)
        shutil.copy2(quant_path, fallback_path)

    return fallback_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compile quantized TFLite model for EdgeTPU")
    parser.add_argument("--quant", type=Path, required=True,
                        help="Path to pv_detector_quant.tflite (from task training:train)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for pv_detector_edgetpu.tflite")
    args = parser.parse_args()

    if not args.quant.exists():
        print(f"ERROR: {args.quant} not found. Run task training:train first.", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    edgetpu_path = compile_edgetpu(args.quant, args.output)

    print(f"\nExport complete:")
    print(f"  EdgeTPU model: {edgetpu_path}")
    labels_src = args.quant.parent / "pv_labels.txt"
    if labels_src.exists() and labels_src != args.output / "pv_labels.txt":
        shutil.copy2(labels_src, args.output / "pv_labels.txt")
        print(f"  Labels:        {args.output / 'pv_labels.txt'}")
    print("\nRun: task training:deploy")


if __name__ == "__main__":
    main()
