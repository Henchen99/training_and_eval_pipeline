from transformers import AutoTokenizer, AutoModelForCausalLM

# Define model name
model_name = "Qwen/Qwen2.5-0.5B"

# Define local directory to save the model
save_directory = "TinyZero/model/Qwen/Qwen2.5-0.5B/"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save model and tokenizer locally
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
