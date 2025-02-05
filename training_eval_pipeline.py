#!/usr/bin/env python
import subprocess
import os
import sys

# Total training steps you wish to run
TOTAL_STEPS = 1000
# Train in increments of 50 steps
STEP_SIZE = 50

# Training script arguments 
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "./data/"
DATASET = "math"
OUTPUT_DIR = "/vault/Tuo/Qwen2.5-Math/evaluation/tiny-r1/src/model_output"
CACHE_DIR = "/vault/Tuo/Qwen2.5-Math/evaluation/tiny-r1/src/cache"

def get_checkpoint_path(current_step: int) -> str:
    model_name = BASE_MODEL.split("/")[-1]
    return os.path.join(OUTPUT_DIR, DATASET, model_name, f"checkpoint-{current_step}")

def run_training(max_steps: int, resume_checkpoint: str = None):
    """
    Runs the training script until max_steps are reached.
    If resume_checkpoint is provided, training will resume from that checkpoint.
    """
    cmd = [
        sys.executable, "train.py",  # call your training file
        "--base_model", BASE_MODEL,
        "--data_path", DATA_PATH,
        "--dataset", DATASET,
        "--output_dir", OUTPUT_DIR,
        "--cache_dir", CACHE_DIR,
        "--max_steps", str(max_steps),
    ]
    if resume_checkpoint:
        cmd.extend(["--resume_from_checkpoint", resume_checkpoint])
    
    print("\n" + "="*40)
    print(f"Running training for max_steps={max_steps}")
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    print(" ".join(cmd))
    
    subprocess.run(cmd, check=True)


def run_evaluation(checkpoint_path: str):
    """
    Runs the evaluation script using the model at checkpoint_path.
    """
    cmd = [
        sys.executable, "eval.py",  # call your evaluation file
        "--model_name_or_path", checkpoint_path,
        # You can add additional evaluation arguments here as needed.
    ]
    
    print("\n" + "="*40)
    print(f"Evaluating model at checkpoint: {checkpoint_path}")
    print(" ".join(cmd))
    
    subprocess.run(cmd, check=True)


def main():
    current_steps = 0
    resume_checkpoint = None  # initially no checkpoint

    while current_steps < TOTAL_STEPS:
        # Increase our training target by STEP_SIZE
        current_steps += STEP_SIZE

        # Run training until current_steps
        run_training(max_steps=current_steps, resume_checkpoint=resume_checkpoint)

        # The training script should have saved a checkpoint at exactly current_steps.
        resume_checkpoint = get_checkpoint_path(current_steps)
        if not os.path.exists(resume_checkpoint):
            print(f"ERROR: Expected checkpoint at {resume_checkpoint} does not exist!")
            sys.exit(1)

        # Run evaluation on the new checkpoint
        run_evaluation(resume_checkpoint)

    print("\nPipeline complete!")

if __name__ == "__main__":
    main()



# mid rounds use 500, final uses 5000