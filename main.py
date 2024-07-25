# 기본
import threading

# GUI 관련
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit, QPushButton

# 커스텀 코드
from utils.sLLM import Ollama_int4_sLLM

class ChatBot(QWidget):
    def __init__(self):
        '''gui 모듈'''
        super().__init__()
        self._init_ui() # user interface 초기화
        self.user_no = 0 # 대화 초기화 할 때 마다 번호가 올라가면서 user_no 형태로 변환
        self.session_id = '' # self.user_no가 섞인 텍스트
        self._init_llm() # llm 초기화
        
    def _init_llm(self):
        '''sLLM 모듈을 불러오고, 멀티턴 대화 기록을 초기화 하며 session_id를 부여'''
        self.llm = Ollama_int4_sLLM(model_name='llama3-ko')
        self._init_chat_session()

    def _init_chat_session(self):
        '''LLM의 멀티턴 기록 초기화'''
        self.session_id = f'user_{self.user_no}'
        self.llm.set_session_id(self.session_id)
        self.user_no += 1

    def _init_ui(self):
        '''UI 초기화'''
        self.setWindowTitle('LlaMa3 8b int4 한국어 On-Device 챗봇')
        self.setGeometry(100, 100, 400, 300)

        # 메인 레이아웃 설정
        self.layout = QVBoxLayout()

        # 출력 영역 (대화 내역)
        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        self.layout.addWidget(self.chat_history)

        # 입력 창
        self.input_text = QLineEdit(self)
        self.input_text.returnPressed.connect(self._send_message)  # Enter 키 이벤트 연결
        self.layout.addWidget(self.input_text)

        # 버튼 레이아웃 설정
        self.button_layout = QHBoxLayout()

        # 전송 버튼
        self.send_button = QPushButton('Send', self)
        self.send_button.clicked.connect(self._send_message)
        self.button_layout.addWidget(self.send_button)

        # 채팅 초기화 버튼
        self.clear_button = QPushButton('채팅 초기화', self)
        self.clear_button.clicked.connect(self._clear_chat)
        self.button_layout.addWidget(self.clear_button)

        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

    def _send_message(self):
        '''gui에서 엔터 혹은 "보내기"버튼을 누르면 텍스트가 위로 옮겨지면서 LLM 추론이 시작됨'''
        user_input = self.input_text.text()
        if user_input:
            # 사용자 입력을 출력 영역에 추가
            self.chat_history.append(f"User: {user_input}")

            # 입력 창을 비우기
            self.input_text.clear()

            # 추론하는데 시간이 걸려 gui가 굳기 때문에 thread로 구현
            threading.Thread(target=self._thread_llm, args=(user_input,)).start() 

    def _thread_llm(self, user_input):
        '''
        gui가 굳지 않도록 thread로 위 '_send_message' 함수에서 실행됨
        
        user_input : LLM에 투입할 텍스트
        '''
        # llm 추론
        response = self.llm.invoke(user_input, self.session_id)

        # PyQt5에서 UI 업데이트는 메인 스레드에서 수행해야 함
        self.chat_history.append(f"ChatBot: {response}")


    def _clear_chat(self):
        '''대화 내역을 초기화하고 '초기화 됨' 메시지를 추가'''
        self.chat_history.clear()
        self._init_chat_session()
        self.chat_history.append("ChatBot: 초기화 됨")

app = QApplication(sys.argv)
chatbot = ChatBot()
chatbot.show()
sys.exit(app.exec_())