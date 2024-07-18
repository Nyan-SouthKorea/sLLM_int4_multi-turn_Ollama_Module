# !CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
# !huggingface-cli download MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M --local-dir='YOUR-LOCAL-FOLDER-PATH'

from llama_cpp import Llama
from transformers import AutoTokenizer

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Llama(
    model_path='llama3_weight/llama3-8b-4bit-q4_1.gguf',
    n_ctx=512,
    n_gpu_layers=-1        # Number of model layers to offload to GPU
)

PROMPT = \
'''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다. 답변은 최대 2문장으로 간결하게 대답하세요'''

instruction = '지금 너무 더워요. 어떻게 하면 좋을까요?'

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize = False,
    add_generation_prompt=True
)

generation_kwargs = {
    "max_tokens":512,
    "stop":["<|eot_id|>"],
    "top_p":0.9,
    "temperature":0.6,
    "echo":True, # Echo the prompt in the output
}

resonse_msg = model(prompt, **generation_kwargs)
print(resonse_msg['choices'][0]['text'][len(prompt):])