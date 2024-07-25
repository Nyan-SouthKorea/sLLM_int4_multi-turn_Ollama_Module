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
        self.response = []
        self.instruct = '''사용자가 질문을 하면 이 질문이 실시간으로 인터넷 검색 등을 해야만 알 수 있는 정보인지 아닌지를 판별해서 대답해 주길 바래.
        예를 들어 오늘이 무슨 요일인지, 날씨가 어떤지, 오늘은 프로 야구에서 어떤 경기를 하는지 등의 인터넷 연결이 있어야지만 대답할 수 있는 질문에 대해서는 너는 반드시 아무 말이나 지어내지 말고, 인터넷이 연결되어 있지 않아서 답변할 수 없다고 말해.
        이런 실시간 인터넷 연결이 필요한 질문에 대해서는 넌 절대 대답을 할 수 없어. 그리고 너가 이미 가지고 있는 상식적인 질문이나 추론적인 질문을 사용자에게 부탁하도록 유도해.
        하지만 라면 레시피를 알려 달라던가, 코드를 만들어 달라던가 등의 실시간 인터넷 연결이 필요 없는 너가 이미 알고 있는 정보로 제공할 수 있는 답변은 성실하게 대답해 주도록 해
        지금 내가 제시한 예시들에만 적용하여 답변을 하지 말고, 사용자의 질문이 인터넷을 요구하는 질문인지 아닌지를 판단하는데 요점을 두도록 해.
        이렇게 해야만 하는 이유는 너는 지금 인터넷에 연결되어 있지 않고, 인터넷 정보가 필요한 대답을 지어서 대답할 경우 잘못된 정보를 사용자에게 제공해서 심각한 손해배성 청구를 받을 수 있기 때문이야.
        내가 지도해준 대로 성실히 일을 수행한다면 정말 큰 보상을 주도록 할게
        너는 이제부터 키즈케어 로봇이야. 너에게 질문을 하는 사용자는 어린 아이고, 너는 가정 안에서 아이의 정서, 안정, 스케줄 등을 케어하는 역할을 하게 될거야.
        이러한 너의 역할을 명시하고 고려하여 답변을 해주길 바래. 어린 아이들은 딱딱하게 대답하지 않고 친구 처럼 반말로 친절하고 장난끼 있게 대답해 주는 것을 좋아해. 절대로 아이에게 존댓말을 하지 말고 반말로 친절하게 대하도록 해. 
        그리고 설명을 생략하지 말고 모두 답변해되, 너무 길게 하진 마.
        '''
        self.remove_word_list = ['[사용자 질문]', '[사용자 질문', '[키즈케어 로봇]', '[키즈케어 로봇']

    def set_session_id(self, session_id):
        '''
        세션 아이디를 dic에 추가해주는 함수. 기존에 존재하는 id를 추가하면 덮어쓰기 됨.

        session_id : 대화 히스토리가 기록되어 있는 세션 ID 입력
        '''
        config = {'configurable': {'session_id': session_id}}
        self.config_dic[session_id] = config
        self.invoke(self.instruct, session_id, instruct_mode=True)

    def invoke(self, human_message, session_id, ai_name='AI 답변', instruct_mode=False):
        '''
        sLLM을 추론하는 함수. session_id의 히스토리에 따라 multi-turn 대화를 한다.

        human_message : sLLM에 입력할 텍스트 전달
        session_id : 이전에 생성한 session_id 전달
        return : 실제 return 값은 없다. 하지만 실시간으로 stream되는 문장을 self.response에 append한다. 답변이 생성되는 동안 TTS를 수행할 수 있다.
        '''
        self.response = []
        sentence = ''
        # response = self.with_message_history.invoke([HumanMessage(content=human_message)], config=self.config_dic[session_id]) # stream 없는 옵션
        if instruct_mode == False: 
            print(f'{ai_name} : ', end='')
        for chunk in self.with_message_history.stream([HumanMessage(content=human_message)], config=self.config_dic[session_id]):
            chunk = chunk.content
            sentence = self._remove_words(f'{sentence}{chunk}')
            # 1개의 문장을 추가하는 기준을 마침표로 본다
            if '.' in chunk:
                self.response.append(deepcopy(sentence))
                sentence = []
            # chunk를 실시간으로 출력한다
            if instruct_mode == False:
                print(self._remove_words(chunk), end='')

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

session_id = 'Ryan'
llm = Ollama_int4_sLLM(model_name='llama3-ko')
llm.set_session_id(session_id)
llm.auto_chatbot(session_id)