import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/qwen2.5_14b_sft_lora"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a cybersecurity expert that has been trained to give precise responses to complex cybersecurity questions. You work in a SOC protecting data for enterprise customers helping to protect their digital assets."},
    {"role": "user", "content": "What is a recommended defense for technique Data Transfer Size Limits (T1030)?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=None,
        top_p=None
    )

result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)