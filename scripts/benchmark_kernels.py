"""Standalone CLI to run custom Metal kernel parity checks & Apple Silicon micro-benchmarks."""

from loopmoe.kernels import run_parity_checks

if __name__ == "__main__":
    run_parity_checks()
