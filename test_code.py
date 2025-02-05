import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set the GPUs to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"  # GPUs 0 and 1
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {num_gpus}")

    # Define device (this will default to the first visible GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Process: Using device {device}")

    # Load the tokenizer
    model_name = "qwen/qwen2.5-14B"  # Replace with your model
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='model',
        device_map="auto"  # Automatically map tokenizer to devices
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model with automatic device mapping
    logger.info("Loading model with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir='model',
        device_map="auto",  # Automatically distribute model across GPUs
        # torch_dtype=torch.float16  # Use bfloat16 for efficient training
    )

    logger.info("Model loaded successfully!")
    # If needed, print out device mapping
    if hasattr(model, 'device_map'):
        logger.info(f"Model is distributed across devices: {model.device_map}")
    else:
        logger.info(f"Model is running on device: {device}")


    # Start a 5-hour timer (optional, can be replaced with actual training loop)
    logger.info("Starting 5-hour timer...")
    total_seconds = 7 * 60 * 60  # 5 hours in seconds
    for _ in tqdm(range(total_seconds), desc="Time remaining", unit="s"):
        time.sleep(1)  # Sleep for 1 second per iteration

    logger.info("7 hours have passed. Timer completed.")

if __name__ == "__main__":
    main()