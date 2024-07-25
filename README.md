# On-Device sLLM Multi-Turn Rag ChatBot

## 서론

sLLM으로 Multi-Turn을 구현하고, RAG를 섞어서 사용자와 자연스러운 대화를 해줄 수 있는 GUI 기반의 챗봇 에이전트를 개발했습니다. 인스트럭션을 통해 인격 또한 부여할 수 있으며, 저의 이름을 사전에 등록하여 "라이언"이라는 아이를 돌보아주는 키즈케어 로봇 페르소나를 적용해 보았습니다.

또한 INT4로 양자화되었기 때문에 테스트 결과 RTX3060에서 ChatGPT-4보다 빠른 추론 속도를 보여줍니다. (Llama 3 8B)  
  
(이미지 클릭 시 영상으로 이동)  
[![alt text](https://img.youtube.com/vi/ScMD64Kkbus/0.jpg)](https://www.youtube.com/watch?v=ScMD64Kkbus)  

## 과정 설명

본 모듈을 개발하는 과정은 아래와 같습니다:

1. sLLM 모델 찾기
2. 한국어로 Fine Tuning
3. fp16 -> int4 양자화
4. 인스트럭션 튜닝
5. Multi-Turn 구축
6. RAG 구축
7. 대화 GUI 모듈 개발
8. 챗봇 에이전트 개발

## 1. sLLM 모델 찾기

오픈소스 LLM은 종류가 정말 다양합니다. 저는 그중 오픈소스의 시작을 알린 Llama를 사용했습니다. 2024년 07월 기준 버전 3.1까지 출시되었고, 작은 sLLM으로 8B 모델을 사용했습니다.

성능은 Llama3가 최고는 아니지만, 유연성이 아주 높은 모델입니다. 8B와 70B는 수치상 크기와 속도가 모두 10배가량 차이가 나는데, 벤치마크 점수는 생각보다 차이가 많이 나지 않습니다.

[Llama3 공식 링크](https://llama.meta.com/)

## 2. 한국어로 Fine Tuning

Llama3 8B 모델의 다국어 점수가 낮기 때문에 한국어로 Fine Tuning이 필요합니다. 저는 이미 한국어로 잘 Fine Tuned 된 모델을 허깅 페이스에서 찾았습니다.

[한국어 Fine Tuned 모델](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)

## 3. FP16 -> INT4 양자화

Floating Point 16 모델을 INT4로 양자화하여 추론 속도를 최대 4배 이상 빠르게 할 수 있습니다. 
Huggingface에 보면 여러 LLM 모델들이 있습니다. 이 모델들은 safetensors로 저장되어 있는 경우가 많습니다. 오늘은 이런 safetensors 모델을 16bit로 gguf 변환을 한 다음에 int4로 양자화하는 과정을 거쳐보도록 하겠습니다. 물론 그 중간에 원본 모델을 추론해 보고, 최종 4bit 모델도 추론해 보도록 하겠습니다.

(gguf : Georgi Gerganov Unified Format의 약자로 오픈소스 모델을 로컬 환경에서 쉽고 빠르게 실행할 수 있는 파일 형식)

[모델 링크](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B/tree/main)

### 내 PC 환경

- 운영체제: Windows 11, WSL2 (Ubuntu 22.04)
- CPU: 13th Gen Intel(R) Core(TM) i7-13700K 3.40 GHz
- GPU: RTX3060
- Python 버전: 3.10 (venv 환경 사용)
- 기타: Anaconda 환경 (conda-forge로 설치한 패키지들)

### float16 모델 추론

먼저 첫 번째로 float16으로 된 모델을 추론해 보도록 하겠습니다.

```python
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
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('')

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
```
이렇게 추론된 결과는 매우 느립니다. 1 토큰 당 0.3초 이상이 소요되며, 이 속도로는 On-Device를 고려할 수 없는 상태입니다.

### 양자화 프로세스
현재 fp16 모델을 int4로 줄여야 합니다. 전체 과정을 한 번에 진행하지는 않았기 때문에 먼저 순서에 대해서 설명하겠습니다. 이 방법이 정답은 아니고 참고한 자료가 있습니다. 양자화 진행 순서는 아래와 같습니다:
  1. safetensors 모델을 gguf로 변환 (여기서는 fp16은 유지됨)  
  2. gguf 모델을 4bit로 양자화 (fp16 -> int4 변환 과정)  

#### 1. safetensors 모델을 gguf로 변환 (fp16 유지)
먼저 모델을 다운로드해야 합니다. 여기에서의 과정은 Anaconda에서 에러가 나기 때문에 wsl2에서 수행했습니다.
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
이후 파이썬을 실행하여 아래와 같이 했을 때 'True'가 출력되면 cuda로 pytorch가 정상적으로 구동되는 것입니다.
```python
import torch
print(torch.cuda.is_available())
```
transformers 라이브러리를 설치합니다.
```bash
pip install transformers
```
이후에 파이썬 파일을 하나 만들어서 아래와 같이 작성하여 hugging face에 있는 모델을 내 로컬 PC에 다운로드합니다.

```python
# 1. 모델 다운로드
from huggingface_hub import snapshot_download
snapshot_download(repo_id='MLP-KTLim/llama-3-Korean-Bllossom-8B', local_dir_use_symlinks=False)
```
이제 모델을 다운로드했습니다. 이후 다운로드한 모델을 GGUF로 변환해야 합니다. 이를 위해 'llama-cpp'라는 레포지토리가 필요합니다.
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```
이제 아래 명령어를 입력하고 모델을 fp16으로 변환합니다.
```bash
python convert_hf_to_gguf.py --outtype f16 --verbose /path/to/your/model
```
예시 명령어는 다음과 같습니다:
```bash
python convert_hf_to_gguf.py --outtype f16 --verbose /home/ubuntu/.cache/huggingface/hub/models--MLP-KTLim--llama-3-Korean-Bllossom-8B/snapshots/8a738f9f622ffc2b0a4a6b81dabbca80406248bf
```
아래 메시지처럼 제대로 변환이 완료되면 안내가 출력됩니다:
```text
INFO:hf-to-gguf:Model successfully exported to /home/ubuntu/.cache/huggingface/hub/models--MLP-KTLim--llama-3-Korean-Bllossom-8B/snapshots/8a738f9f622ffc2b0a4a6b81dabbca80406248bf/ggml-model-f16.gguf
```

#### 2. gguf 모델을 4bit로 양자화 (fp16 -> int4 변환 과정)
이제 fp16으로 변환된 gguf 파일을 int4로 양자화해보겠습니다. 아까의 github releases에 들어가서 'llama-b3369-bin-ubuntu-x64.zip'을 다운로드하고 wsl2 디렉터리 안에 옮겨놓습니다.

```bash
unzip llama-b3376-bin-ubuntu-x64.zip
```
아래의 명령어를 입력하여 양자화합니다:
```bash
./build/bin/llama-quantize /path/to/your/gguf-model gguf-model-4bit.gguf Q4_1
```
이제 모델 양자화가 성공적으로 완료되었습니다.

### 4bit로 양자화된 llama3 gguf 모델 인퍼런스 - Ollama 방식
이제 wsl2에 가상환경을 하나 더 새로 만들고 ollama를 설치한 다음 아래와 같이 실행하면 4bit로 양자화된 모델을 추론해 볼 수 있습니다.

```bash
touch Modelfile
```
VS 코드 등의 에디터에서 아래와 같이 내용을 작성합니다:

```text
FROM Llama-3-Open-Ko-8B-Q8_0.gguf

TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

PARAMETER temperature 0
PARAMETER num_predict 3000
PARAMETER num_ctx 4096
PARAMETER stop <s>
PARAMETER stop </s>
```
Ollama가 설치된 디바이스에서 아래와 같이 입력하여 Ollama 모델을 생성합니다:

```bash
ollama create llama3-ko -f Modelfile
```
이제 custom 모델을 실행해 볼 수 있습니다:

```bash
ollama run llama3-ko
```


## 4. 인스트럭션 튜닝

다음과 같이 인스트럭션을 설정하여 사용자의 정보를 기반으로 맞춤형 대화를 할 수 있도록 했습니다.

```python
def _get_instruct(self):
    file_list = ['base', 'few_shot', 'informations']
    path = 'instruct'
    instruction_template = ''
    for file in file_list:
        with open(f'{path}/{file}.txt', 'r', encoding='utf-8-sig') as f:
            full_txt = f.read()
        instruction_template = f'{instruction_template}\n{full_txt}'
    return instruction_template
```

## 5. Multi-Turn 구축
LangChain을 통해 Multi-Turn 대화와 RAG를 통합하여 하나의 모듈로 구현했습니다.
[LangChain 공식 사이트](https://www.langchain.com/)

(이미지 클릭 시 영상으로 이동)  
[![alt text](https://img.youtube.com/vi/sL4Q9bR2FGs/0.jpg)](https://www.youtube.com/watch?v=sL4Q9bR2FGs)  

```python
from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

self.with_message_history = RunnableWithMessageHistory(self.model, self._get_session_history)
response = self.with_message_history.invoke([HumanMessage(content=human_message)], config=self.config_dic[session_id])
```

## 6. RAG 구축
RAG 시스템을 사용하여 정보들을 벡터화하고, 검색 및 증강하여 답변을 생성합니다.

(이미지 클릭 시 영상으로 이동)  
[![alt text](https://img.youtube.com/vi/e-kZljGkFs0/0.jpg)](https://www.youtube.com/watch?v=e-kZljGkFs0)  


```python
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain

# 텍스트 파일 로드 및 전처리
text_file_path = "for_rag.txt"
documents = TextLoader(text_file_path, encoding='utf-8').load()

# 문서 청크로 분할
def split_docs(documents, chunk_size=5000, chunk_overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

# 벡터 저장
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

# Q&A 체인 설정
llm = ChatOllama(model="llama3-ko", streaming=True)
qa_chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
```

## 7. 대화 GUI 모듈 개발
PyQT5를 사용하여 간단한 GUI를 개발했습니다.

(이미지 클릭 시 영상으로 이동)  
[![alt text](https://img.youtube.com/vi/44JFK6VS1aE/0.jpg)](https://www.youtube.com/watch?v=44JFK6VS1aE)  


```python
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit, QPushButton

class ChatBot(QWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()
        self.user_no = 0
        self.session_id = ''
        self._init_llm()
        
    def _init_llm(self):
        self.llm = Ollama_int4_sLLM(model_name='llama3-ko')
        self._init_chat_session()

    def _init_chat_session(self):
        self.session_id = f'user_{self.user_no}'
        self.llm.set_session_id(self.session_id)
        self.user_no += 1

    def _init_ui(self):
        self.setWindowTitle('LlaMa3 8b int4 한국어 On-Device 챗봇')
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        self.layout.addWidget(self.chat_history)

        self.input_text = QLineEdit(self)
        self.input_text.returnPressed.connect(self._send_message)
        self.layout.addWidget(self.input_text)

        self.button_layout = QHBoxLayout()

        self.send_button = QPushButton('Send', self)
        self.send_button.clicked.connect(self._send_message)
        self.button_layout.addWidget(self.send_button)

        self.clear_button = QPushButton('채팅 초기화', self)
        self.clear_button.clicked.connect(self._clear_chat)
        self.button_layout.addWidget(self.clear_button)

        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

    def _send_message(self):
        user_input = self.input_text.text()
        if user_input:
            self.chat_history.append(f"User: {user_input}")
            self.input_text.clear()
            threading.Thread(target=self._thread_llm, args=(user_input,)).start() 

    def _thread_llm(self, user_input):
        response = self.llm.invoke(user_input, self.session_id)
        self.chat_history.append(f"ChatBot: {response}")

    def _clear_chat(self):
        self.chat_history.clear()
        self._init_chat_session()
        self.chat_history.append("ChatBot: 초기화 됨")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    chatbot = ChatBot()
    chatbot.show()
    sys.exit(app.exec_())
```

## 기타. RAG vs Instruction
사용자 정보는 RAG 방식으로 관리하는 것이 대화의 지속성과 정보의 유지 측면에서 더 유리합니다.  

## 향후 계획
이번 연구를 통해 sLLM으로 Multi-Turn을 구현하고, RAG를 통합하여 사용자와 자연스러운 대화를 할 수 있는 GUI 기반의 챗봇 에이전트를 성공적으로 개발하였습니다.

향후 STT와 TTS 기술을 융합하여 음성 대화형 에이전트를 개발할 예정이며, Jetson 등의 싱글 보드 디바이스에서 테스트를 통해 경량화된 모델을 서비스 로봇에 탑재하여 사용자와 소통하는 것이 목표입니다.

카메라를 활용한 영상 분석 AI 혹은 이미지 분석 모델을 적용하여 VLM 형태의 시스템도 개발해 보고 싶습니다.

궁금한 사항은 댓글 언제나 환영입니다

[네이버 블로그](https://blog.naver.com/112fkdldjs/223524280555)
