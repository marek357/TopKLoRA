#!/usr/bin/env python3
"""
Run Hydra sweeps in parallel across multiple GPUs.

Usage examples:
---------------
# SFT sweep on GPU 0 and 1 in parallel (splits the sweep space)
python scripts/run_sweep.py --gpus 0,1 \
    --r 1024,4096,8192 \
    --k 128,256,512 \
    --k_final 8,32 \
    --module_type mlp,mlp_attn

# DPO sweep (use --mode dpo)
python scripts/run_sweep.py --mode dpo --gpus 0,1 \
    --r 1024,4096 \
    --k 128,256 \
    --module_type mlp,mlp_attn

# Run specific configs on specific GPUs
python scripts/run_sweep.py --gpu 0 --r 1024,4096 --k 128 --module_type mlp &
python scripts/run_sweep.py --gpu 1 --r 1024,4096 --k 128 --module_type mlp_attn &

# Dry run to see what would be executed
python scripts/run_sweep.py --dry-run --gpus 0,1 --r 1024,4096 --k 128,256

# Use parallel mode for multiprocessing execution
python scripts/run_sweep.py --gpus 0,1,2,3 --parallel --r 1024,2048,4096,8192 --k 64,128,256,512
"""

import argparse
import itertools
import os
import subprocess
import sys
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SweepConfig:
    r: List[int]
    k: List[int]
    k_final: List[int]
    module_type: List[str]
    layer: List[int]
    extra_args: List[str]
    mode: str  # 'sft' or 'dpo'


def parse_int_list(s: str) -> List[int]:
    """Parse comma-separated integers."""
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",")]


def parse_str_list(s: str) -> List[str]:
    """Parse comma-separated strings."""
    if not s:
        return []
    return [x.strip() for x in s.split(",")]


def generate_configs(sweep: SweepConfig) -> List[dict]:
    """Generate all combinations of sweep parameters."""
    configs = []
    for r, k, k_final, module_type, layer in itertools.product(
        sweep.r, sweep.k, sweep.k_final, sweep.module_type, sweep.layer
    ):
        # Skip invalid combinations (k > r or k_final > k)
        if k > r:
            continue
        if k_final > k:
            continue
        configs.append(
            {
                "r": r,
                "k": k,
                "k_final": k_final,
                "module_type": module_type,
                "layer": layer,
            }
        )
    return configs


def build_command(
    config: dict,
    gpu: int,
    extra_args: List[str],
    mode: str = "sft",
    python_bin: str = "python",
    bs_r8192: int | None = None,
    ga_r8192: int | None = None,
) -> str:
    """Build the training command for a single config."""
    if mode == "dpo":
        training_config = "dpo_topk_sweep"
        experiment_path = "training.dpo_experiment.lora"
    else:
        training_config = "sft_recommended_topk_sweep"
        experiment_path = "training.sft_experiment.lora"

    cmd_parts = [
        f"CUDA_VISIBLE_DEVICES={gpu}",
        python_bin,
        "main.py",
        f"training={training_config}",
        f"{experiment_path}.r={config['r']}",
        f"{experiment_path}.k={config['k']}",
        f"{experiment_path}.k_final={config['k_final']}",
        f"{experiment_path}.module_type={config['module_type']}",
        f"{experiment_path}.layer={config['layer']}",
    ]

    # Optional automatic batch-size override for large r to avoid OOM (e.g., r=8192 on A40)
    if config["r"] >= 8192 and mode == "dpo":
        if bs_r8192:
            cmd_parts.append(f"training.dpo.per_device_train_batch_size={bs_r8192}")
            cmd_parts.append(f"training.dpo.per_device_eval_batch_size={max(1, bs_r8192//2)}")
        if ga_r8192:
            cmd_parts.append(f"training.dpo.gradient_accumulation_steps={ga_r8192}")

    cmd_parts.extend(extra_args)
    return " ".join(cmd_parts)


def split_configs_across_gpus(
    configs: List[dict], gpus: List[int]
) -> List[Tuple[int, List[dict]]]:
    """Split configs evenly across GPUs."""
    n_gpus = len(gpus)
    splits = [[] for _ in range(n_gpus)]
    for i, config in enumerate(configs):
        splits[i % n_gpus].append(config)
    return [(gpu, split) for gpu, split in zip(gpus, splits) if split]


def run_sequential(
    configs: List[dict],
    gpu: int,
    extra_args: List[str],
    mode: str = "sft",
    dry_run: bool = False,
    bs_r8192: int | None = None,
    ga_r8192: int | None = None,
):
    """Run configs sequentially on a single GPU."""
    for i, config in enumerate(configs):
        cmd = build_command(config, gpu, extra_args, mode, bs_r8192=bs_r8192, ga_r8192=ga_r8192)
        print(
            f"\n[GPU {gpu}] Running {i + 1}/{len(configs)}: r={config['r']}, k={config['k']}, k_final={config['k_final']}, {config['module_type']}"
        )
        print(f"  Command: {cmd}")

        if not dry_run:
            result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent.parent)
            if result.returncode != 0:
                print(
                    f"  [WARNING] Command failed with return code {result.returncode}"
                )


def run_parallel_multiprocess(
    gpu_splits: List[Tuple[int, List[dict]]],
    extra_args: List[str],
    mode: str = "sft",
    dry_run: bool = False,
    bs_r8192: int | None = None,
    ga_r8192: int | None = None,
):
    """Run sweeps in parallel using multiprocessing."""
    import multiprocessing as mp

    def worker(gpu: int, configs: List[dict]):
        run_sequential(configs, gpu, extra_args, mode, dry_run, bs_r8192=bs_r8192, ga_r8192=ga_r8192)

    processes = []
    for gpu, configs in gpu_splits:
        p = mp.Process(target=worker, args=(gpu, configs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def generate_shell_scripts(
    gpu_splits: List[Tuple[int, List[dict]]],
    extra_args: List[str],
    mode: str = "sft",
    output_dir: str = "sweep_scripts",
    conda_env: str = "klora",
    bs_r8192: int | None = None,
    ga_r8192: int | None = None,
):
    """Generate shell scripts for each GPU to run manually or via tmux."""
    os.makedirs(output_dir, exist_ok=True)

    python_bin = str(Path.home() / "miniconda3" / "envs" / "klora" / "bin" / "python3")

    master_script = f"#!/bin/bash\n# Master script to launch all GPU sweeps in tmux ({mode.upper()} mode)\n\n"

    for gpu, configs in gpu_splits:
        script_path = os.path.join(output_dir, f"sweep_gpu{gpu}.sh")
        with open(script_path, "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(
                f"# {mode.upper()} Sweep script for GPU {gpu} - {len(configs)} experiments\n"
            )
            f.write(f"# Generated by run_sweep.py\n\n")
            f.write(f"set -e  # Exit on error\n\n")
            f.write(f"cd {Path(__file__).parent.parent}\n")
            f.write(f"source ~/.bashrc\n")
            f.write(f'PYTHON_BIN="{python_bin}"\n')
            f.write(f'echo "Using python: $PYTHON_BIN"\n\n')
            f.write(f"export CUDA_VISIBLE_DEVICES={gpu}\n\n")

            for i, config in enumerate(configs):
                cmd = build_command(
                    config,
                    gpu,
                    extra_args,
                    mode,
                    python_bin=python_bin,
                    bs_r8192=bs_r8192,
                    ga_r8192=ga_r8192,
                ).replace(f"CUDA_VISIBLE_DEVICES={gpu} ", "")
                f.write(
                    f"echo '=== [{i + 1}/{len(configs)}] r={config['r']}, k={config['k']}, k_final={config['k_final']}, {config['module_type']} ==='\n"
                )
                f.write(f"{cmd}\n\n")

            f.write(f"echo 'GPU {gpu} {mode.upper()} sweep complete!'\n")

        os.chmod(script_path, 0o755)
        print(f"Generated: {script_path} ({len(configs)} experiments)")

        master_script += f"tmux new-window -n 'gpu{gpu}' 'bash {script_path}'\n"

    master_path = os.path.join(output_dir, "launch_all.sh")
    with open(master_path, "w") as f:
        f.write(master_script)
    os.chmod(master_path, 0o755)
    print(f"\nGenerated master launch script: {master_path}")
    print(f"Run with: tmux && bash {master_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run TopK LoRA sweeps across multiple GPUs"
    )

    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        default="dpo",
        choices=["sft", "dpo"],
        help="Training mode: 'sft' for supervised fine-tuning, 'dpo' for direct preference optimization",
    )

    # GPU selection
    parser.add_argument("--gpu", type=int, help="Single GPU to use")
    parser.add_argument(
        "--gpus", type=str, help="Comma-separated list of GPUs (e.g., '0,1,2,3')"
    )

    # Sweep parameters
    parser.add_argument(
        "--r", type=str, default="4096", help="Comma-separated r values"
    )
    parser.add_argument("--k", type=str, default="512", help="Comma-separated k values")
    parser.add_argument(
        "--k_final", type=str, default="32", help="Comma-separated k_final values"
    )
    parser.add_argument(
        "--module_type",
        type=str,
        default="mlp",
        help="Comma-separated: mlp, mlp_attn, attn",
    )
    parser.add_argument(
        "--layer", type=str, default="18", help="Comma-separated layer values"
    )

    # Execution options
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run GPUs in parallel (default: generate scripts)",
    )
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Generate shell scripts for manual/tmux execution",
    )
    parser.add_argument(
        "--scripts-dir",
        type=str,
        default="sweep_scripts",
        help="Directory for generated scripts",
    )
    parser.add_argument(
        "--conda-env", type=str, default="klora", help="Conda environment name"
    )

    # Optional: override batch size / grad accumulation for large r (OOM mitigation)
    parser.add_argument(
        "--bs-r8192",
        type=int,
        default=None,
        help="If set, overrides training.dpo.per_device_train_batch_size when r>=8192",
    )
    parser.add_argument(
        "--ga-r8192",
        type=int,
        default=None,
        help="If set, overrides training.dpo.gradient_accumulation_steps when r>=8192",
    )

    # Extra args to pass to main.py
    parser.add_argument(
        "extra", nargs="*", help="Additional arguments to pass to main.py"
    )

    args = parser.parse_args()

    # Determine GPUs
    if args.gpu is not None:
        gpus = [args.gpu]
    elif args.gpus:
        gpus = parse_int_list(args.gpus)
    else:
        gpus = [0]

    # Build sweep config
    sweep = SweepConfig(
        r=parse_int_list(args.r),
        k=parse_int_list(args.k),
        k_final=parse_int_list(args.k_final),
        module_type=parse_str_list(args.module_type),
        layer=parse_int_list(args.layer),
        extra_args=args.extra or [],
        mode=args.mode,
    )

    # Generate all configs
    configs = generate_configs(sweep)
    print(f"Generated {len(configs)} {args.mode.upper()} experiment configurations")
    print(f"Using GPUs: {gpus}")

    if len(configs) == 0:
        print("No valid configurations generated!")
        sys.exit(1)

    # Split across GPUs
    gpu_splits = split_configs_across_gpus(configs, gpus)

    for gpu, split_configs in gpu_splits:
        print(f"  GPU {gpu}: {len(split_configs)} experiments")

    # Execute or generate scripts
    if args.scripts:
        generate_shell_scripts(
            gpu_splits,
            sweep.extra_args,
            args.mode,
            args.scripts_dir,
            args.conda_env,
            bs_r8192=args.bs_r8192,
            ga_r8192=args.ga_r8192,
        )
    elif args.parallel and len(gpus) > 1:
        print(f"\nRunning {args.mode.upper()} in parallel across GPUs...")
        run_parallel_multiprocess(
            gpu_splits,
            sweep.extra_args,
            args.mode,
            args.dry_run,
            bs_r8192=args.bs_r8192,
            ga_r8192=args.ga_r8192,
        )
    else:
        # Sequential on each GPU (useful for single GPU or dry-run)
        for gpu, split_configs in gpu_splits:
            run_sequential(
                split_configs,
                gpu,
                sweep.extra_args,
                args.mode,
                args.dry_run,
                bs_r8192=args.bs_r8192,
                ga_r8192=args.ga_r8192,
            )


if __name__ == "__main__":
    main()
