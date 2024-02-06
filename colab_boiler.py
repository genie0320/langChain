# __________ install ___________
# !pip install --quiet icecream
# !pip install --quiet langchain langchain_openai
# ______________________________

import pprint
from icecream import ic
from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()


# pretty print setting
def pp(object):
    ppr = pprint.PrettyPrinter(
        # indent=40,
        width=80
    )
    return ppr.pprint(object)


# set LLM
def ask_openai(*query):
    llm = OpenAI(
        # api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=256,
        verbose=True,
        # model = model
    )
    if query:
        return llm.invoke(str(query))
    else:
        return llm


ic(ask_openai("what is your name?"))

# pure openai setting
# llm = OpenAI(
#     # api_key=OPENAI_API_KEY,
#     temperature=0,
#     max_tokens=256,
#     verbose=True,
#     # model = model
# )
# ic(llm.invoke("hello?"))
