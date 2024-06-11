from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
token="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"

model_path = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
     
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))

else:
    
    device = torch.device("cpu")
    print("Device: ", device)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    trust_remote_code=True,
    torch_dtype = torch.bfloat16,
)
model.to(device)
    

tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, trust_remote_code=True)
     

messages = [
    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "who developed you?"},
]
     

input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
# input_ids
     

gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.6,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)