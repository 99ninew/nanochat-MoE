import os
import subprocess
import itertools
import sys

# -----------------------------------------------------------------------------
# Hyperparameter Search Space
# default:
# embedding_lr = 0.2
# matrix_lr = 0.02
search_space = {
    # "learning_rate": [0.01, 0.005, 0.001]
    "learning_rate": [3e-4, 1e-4, 8e-5]
    # "matrix_lr": [0.01, 0.02, 0.04],
    # "embedding_lr": [0.1, 0.2],
    # "matrix_lr": [0.005, 0.01],
    # "embedding_lr": [0.01, 0.05, 0.1],
    # "unembedding_lr": [0.004],
    # "init_lr_frac": [0.5, 1.0],
}

# Fixed parameters for all runs
# These match the settings in speedrun_moe.sh for midtraining
fixed_args = [
    "--device_batch_size=8",
    "--max_seq_len=1024",
    "--total_batch_size=524288",
    # You might want to limit iterations for a sweep to save time
    # "--num_iterations=1000", 
]

# -----------------------------------------------------------------------------

def get_nproc():
    # Try to detect number of GPUs
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 1

def main():
    # Detect number of GPUs
    nproc = get_nproc()
    if nproc == 0:
        nproc = 1
    print(f"Detected {nproc} GPUs")

    # Generate all combinations of hyperparameters
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))

    print(f"Found {len(combinations)} combinations to sweep.")

    for i, combination in enumerate(combinations):
        print(f"\n--- Starting run {i+1}/{len(combinations)} ---")
        
        # Construct arguments for this run
        run_args = []
        run_name_parts = ["mid_sweep"]
        
        for key, value in zip(keys, combination):
            run_args.append(f"--{key}={value}")
            run_name_parts.append(f"{key}{value}")
        
        run_name = "_".join(run_name_parts)
        run_args.append(f"--run={run_name}")

        # Construct the full command
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            "-m", "scripts.mid_train",
            "--"
        ] + run_args + fixed_args

        cmd_str = " ".join(cmd)
        print(f"Executing: {cmd_str}")

        try:
            # Run the command
            # We use env=os.environ to pass through environment variables like WANDB_API_KEY
            subprocess.run(cmd, check=True, env=os.environ)
        except subprocess.CalledProcessError as e:
            print(f"Run {run_name} failed with error: {e}")
            # Decide whether to continue or stop. Here we continue to the next run.
            continue
        except KeyboardInterrupt:
            print("\nSweep interrupted by user.")
            sys.exit(1)

    print("\nSweep completed.")

if __name__ == "__main__":
    main()
