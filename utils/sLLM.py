# 기본
from copy import deepcopy

# 추가
from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage


class Ollama_int4_sLLM:
    def __init__(self, model_name):
        self.model = ChatOllama(model=model_name)
        self.store = {}
        self.with_message_history = RunnableWithMessageHistory(self.model, self._get_session_history)
        self.config_dic = {}
        self.instruct = self._get_instruct()
        self.remove_word_list = ['[사용자 질문]', '[사용자 질문', '[키즈케어 로봇]', '[키즈케어 로봇']

    def _get_instruct(self):
        '''
        LLM에 입력되는 인스트럭션 텍스트를 불러옴
        
        return : 인스트럭션 텍스트
        '''
        file_list = ['base', 'few_shot', 'informations'] # 순서대로 읽어드릴 텍스트 파일 이름들
        path = 'instruct'
        instruction_template = ''
        for file in file_list:
            with open(f'{path}/{file}.txt', 'r', encoding='utf-8-sig') as f:
                full_txt = f.read()
            instruction_template = f'{instruction_template}\n{full_txt}'
        return instruction_template

    def set_session_id(self, session_id):
        '''
        세션 아이디를 dic에 추가해주는 함수. 기존에 존재하는 id를 추가하면 덮어쓰기 됨.

        session_id : 대화 히스토리가 기록되어 있는 세션 ID 입력
        '''
        config = {'configurable': {'session_id': session_id}}
        self.config_dic[session_id] = config
        self.invoke(self.instruct, session_id, instruct_mode=True)

    def invoke(self, human_message, session_id, instruct_mode=False):
        '''
        sLLM을 추론하는 함수. session_id의 히스토리에 따라 multi-turn 대화를 한다.

        human_message : sLLM에 입력할 텍스트 전달
        session_id : 이전에 생성한 session_id 전달
        return : LLM이 생성한 답변 반환
        '''
        response = self.with_message_history.invoke([HumanMessage(content=human_message)], config=self.config_dic[session_id]) # stream 없는 옵션
        return response.content

    def auto_chatbot(self, session_id):
        '''
        자동으로 챗봇 대화를 시작하는 함수. 시작 전 모델과 session id가 정의되어 있어야 한다.

        session_id : 대화 히스토리를 쌓을 session_id를 입력
        '''
        print('대화 종료를 위해서는 "exit()"를 입력하시오')
        print('대화를 시작해주세요.')
        while True:
            human_message = input(f'\n{session_id} : ')
            if human_message == 'exit()':
                print('대화를 종료합니다')
                break
            self.invoke(human_message, session_id)
            
    def _get_session_history(self, session_id):
        '''
        세션 아이디에 따른 대화 히스토리를 가져오는 함수. 
        세션 아이디가 존재하지 않으면 새로운 대화 히스토리를 생성하여 저장하고 반환함.

        session_id : 대화 히스토리가 기록되어 있는 세션 ID 입력
        return : session_id에 해당하는 InMemoryChatMessageHistory 객체 반환
        '''
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def _remove_words(self, txt):
        for word in self.remove_word_list:
            # 해당 단어를 삭제하고, 이중 공백이 있을 경우 정리(단어가 없어도 에러가 발생하진 않는다)
            txt = txt.replace(word, '').replace('  ', ' ')
        return txt