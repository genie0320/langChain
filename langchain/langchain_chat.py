from dotenv import load_dotenv
import pprint
from icecream import ic
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

user_input = 'Do you know korea?'
MODEL="local-model"

# Point to the local server
local_llm = OpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="not-needed",
    temperature=0,
    max_tokens=256,
    streaming=True,
    )

local_chat_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="not-needed",
    temperature=0,
    max_tokens=256,
    streaming=True,
    )

open_llm = OpenAI(
    temperature=0,
    max_tokens=256,
    streaming=True,    
)
open_chat_llm = ChatOpenAI(
    temperature=0,
    max_tokens=256,
    streaming=True,    
)

template = PromptTemplate.from_template('''
    Answer to users question below with easy simple sentence.:
    
    [questions]
    {user_input}
    
    
    '''
    )

chat_template = ChatPromptTemplate.from_template('''
    Answer to users question below with easy simple sentence.:
    
    [questions]
    {user_input}
    
    
    '''
)

prompt = template.format_prompt(user_input = user_input)
ic(prompt)
chat_prompt = chat_template.format_prompt(user_input = user_input)
ic(prompt)

# ic.disable()
# # Answer
# local_res = local_llm.invoke(prompt)
# pprint.pprint(f'local_res : {local_res}')

# open_res = open_llm.invoke(prompt)
# pprint.pprint(f'open_res : {open_res}')

# # Stream answer
# # callbacks를 사용하지 않아도 stream이 된다. 
# for chunk in local_llm.stream(prompt):
#     print(chunk, end="", flush=True)












# user_input = 'do you know why TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF model is always create dummy chat history which I did not input?'
# llm.invoke(user_input)