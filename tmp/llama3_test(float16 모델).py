import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

# 명시적으로 pad_token_id를 eos_token_id로 설정
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
instruction = "난 지금 너무 슬퍼. 기분 좋은 이야기 해줄 수 있어?"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True,  # Ensure padding is applied
    truncation=True  # Ensure truncation if necessary
).to(model.device)

# Create attention mask
attention_mask = (input_ids != tokenizer.pad_token_id).long()

# Ensure the eos_token_id is valid
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')

terminators = [
    tokenizer.eos_token_id
]

# Start with the initial input_ids
current_input_ids = input_ids


start = time.time()
# Loop to generate tokens one by one
for _ in range(2048):  # Maximum number of tokens to generate
    outputs = model.generate(
        current_input_ids,
        max_new_tokens=1,  # Generate one token at a time
        eos_token_id=terminators[0],  # Use the first (and only) terminator
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        attention_mask=attention_mask,  # Pass the attention mask
        pad_token_id=tokenizer.pad_token_id  # Explicitly set pad_token_id
    )
    
    # Get the newly generated token (last token in the sequence)
    next_token_id = outputs[0, -1].unsqueeze(0)
    
    # Decode and print the new token
    decoded_word = tokenizer.decode(next_token_id, skip_special_tokens=True)
    if decoded_word:
        print(decoded_word, end='', flush=True)
    
    # Check if the generated token is the EOS token
    if next_token_id.item() == tokenizer.eos_token_id:
        break
    
    # Append the new token to the current input_ids
    current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=-1)
    # Update the attention mask for the new input
    attention_mask = torch.cat([attention_mask, torch.tensor([[1]], device=model.device)], dim=-1)

spent = time.time() - start
print(f'시간 측정: {int(spent)}')
