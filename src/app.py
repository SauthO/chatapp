import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder


def create_agent_chain():
    chat = ChatOpenAI(
        model_name = "gpt-4o-mini-2024-07-18",
        temperature = 0.5,
        streaming = True,
        )
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(tools, chat, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory,)


st.title("langchain-streamlit-app")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")
print(prompt)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        if "agent_chain" not in st.session_state:
            st.session_state.agent_chain = create_agent_chain()
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
