import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

import streamlit as st
from langchain.memory import StreamlitChatMessageHistory
import time 
import tiktoken
from loguru import logger 

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    # unstructured_directory_loader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.callbacks import get_openai_callback

# TODO: Change to auto tokenizer
def token_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(files):

    docs_list=[]

    for file in files:
        file_name = file.name

        with open(file_name, 'wb') as f:
            f.write(file.getvalue())
            logger.info(f'Uploaded {file_name}') # 여기 logger 부분은 프로세스 안내를 위해서 채팅창에 나오도록 하자.

            # Load file and Split by "page"
            if '.pdf' in file_name:
                loader = PyPDFLoader(file_name)
            
            elif '.docx' in file_name:
                loader = Docx2txtLoader(file_name)
            
            elif '.pptx' in file_name:
                loader = UnstructuredPowerPointLoader(file_name)
            
            elif '.txt' in file_name:
                loader = TextLoader(file_name)
            
            else :
                with st.sidebar:
                    st.error(f'File type is not supported : {file_name}')
            
            pages = loader.load_and_split()

            docs_list.extend(pages)
        
        with st.sidebar:
            st.write(len(docs_list))

    return docs_list

def get_chunks(docs_list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap = 90,
        length_function = token_len
    )

    chunks = splitter.split_documents(docs_list)
    return chunks

def get_db(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}        
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db

def get_answer(db, openai_api_key):
    openai_chat =ChatOpenAI(
        openai_api_key=openai_api_key, 
        model_name = 'gpt-3.5-turbo',
        temperature=0
    )
    chat_memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=openai_chat, 
            chain_type="stuff", 
            retriever=db.as_retriever(search_type = 'mmr', vervose = True), 
            memory=chat_memory,
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain
def main():
    st.set_page_config(
        page_title = 'Dirchat',
        page_icon = ':books:'
    )

    st.title('_Private Data :red[QA chat]_:books:')

    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = None

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = None

    # if "processComplete" not in st.session_state:
    #     st.session_state.processComplete = None

    with st.sidebar:
        if OPENAI_API_KEY:
            openai_api_key = OPENAI_API_KEY
        else :
            openai_api_key = st.text_input('OpenAI API key',key='chatbot_api_key', type = 'password')
        added_files = st.file_uploader('Upload your file', type = ['pdf','txt', 'docx', 'pptx'], accept_multiple_files=True)
        process = st.button('Process')
        clear = st.button('Clear')
    if clear :
        del st.session_state['messages']
    if process:
        if not openai_api_key:
            st.info('궁금하면 복채를 먼저 얹어봐.')
            st.stop() # 잠시멈춤
        pages = get_text(added_files)
        chunks = get_chunks(pages)
        db = get_db(chunks)

        st.session_state.conversation = get_answer(db,openai_api_key)

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {
                'role':'assistant',
                'content':'what can I do help you?'
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # history = StreamlitChatMessageHistory(key='chat_messages')


    query = st.chat_input('질문을 입력해주세요')

    if query:
        st.session_state.messages.append({'role':'user', 'content':query})

        with st.chat_message('user'):
            st.markdown(query)

        with st.chat_message('assistant'):
            chain = st.session_state.conversation

            with st.spinner('processing...'):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']
                st.markdown(response)
                with st.sidebar:
                    st.write(st.session_state.messages)

                with st.expander("참고 문서 확인"):
                    for src in source_documents:
                        st.markdown(src.page_content[:200], help = src.metadata['source'])


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()