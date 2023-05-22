# from transformers import AutoModelForSeq2SeqLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel, PeftConfig
import torch

peft_model_id = "lora-alpaca"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = model.to("cuda")
model.eval()
inputs = tokenizer("Is this contaminated? 172 119 4.4 4.1 1.5 1.2 2.51 2.36", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"].to("cuda"),max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])