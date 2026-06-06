from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from agent_tools import search_faq, reformulate_query

load_dotenv()


def get_llm_model()->BaseChatModel:
    groq= ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    open_ai=ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )
    return open_ai

def get_agent():
    system_prompt="""
    You are a helpful FAQ assistant with access to a knowledge base.
    Your goal is to nswer user questions accurately using the available tools.
    
    Guidelines:
    1. Start by using the search_faq tool to find relevant information.
    2. if the query is complex use reformulate_query to search  different aspects.
    3. Always provide a clear and concise answer bsed on the retrieved information
    4. If you can not find relevant information, clearly state that.
    
    Think step by step and use tolls strategically to provide the answer. """

    tools=[search_faq,reformulate_query]

    agent=create_agent(
        model=get_llm_model(),
        tools=tools,
        system_prompt=system_prompt,

    )
    return agent

if __name__ == "__main__":
    agent_message={"messages":[("human","Explain roaming activation")]}
    result=get_agent().invoke(agent_message)
    print(result['messages'][-1].content)
