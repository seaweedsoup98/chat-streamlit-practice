import os
import streamlit as st
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["LANGCHAIN_TRACING"] = "false"

def get_secret(key: str) -> str:
    '''secret key 가져오는 함수'''
    if "ST_SECRETS" in os.environ:  # Streamlit Cloud 환경
        return st.secrets[key]
    else:  # 로컬 환경
        return os.getenv(key)

def create_agent_chain(history):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )
    
    tools = load_tools(["ddg-search", "wikipedia"])
    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )
    
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory)


if __name__ == "__main__":
    st.title("Langchain Chat Demo")
    
    history = StreamlitChatMessageHistory()
    
    for message in history.messages:
        st.chat_message(message.type).write(message.content)
    
    prompt = st.chat_input("무엇을 도와드릴까요?")
    
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            callback = StreamlitCallbackHandler(st.container())
            agent_chain = create_agent_chain(history)
            response = agent_chain.invoke(
                {"input": prompt},
                {"callbacks": [callback]},
            )
            
            st.markdown(response["output"])

