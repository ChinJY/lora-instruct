import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
    # DataCollatorForSeq2Seq,
    # PreTrainedModel,
    # PreTrainedTokenizer,
    # Trainer,
    # TrainingArguments
)

# Set the path to your local model
local_model_path = "lora-alpaca/runs/May21_09-41-23_nf5tqxdm40"

# Check if the path exists
if os.path.exists(local_model_path):
    # Load the tokenizer and model from the local path
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
else:
    print(f"The model path '{local_model_path}' does not exist.")

# Tokenize and encode input text
input_text = "This is an example sentence."
encoded_input = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    return_tensors="pt",
)

# Perform a forward pass through the model
output = model(**encoded_input)

# Process the output
logits = output.logits

print(output)