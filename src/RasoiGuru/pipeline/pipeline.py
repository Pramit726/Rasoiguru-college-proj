from typing import List
from src.RasoiGuru.components.create_tools import ToolCreator
from src.RasoiGuru.components.generation import Generator
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor

def create_pipeline(vectorstores: List, memory: ConversationBufferWindowMemory) -> AgentExecutor:
    """
    Creates a pipeline for generating responses.

    Args:
        vectorstores: A list of PineconeVectorStore objects.

    Returns:
        A tuple containing the created tools and the agent executor.
    """
    tool_creator = ToolCreator()
    retrievers = tool_creator.create_retriever(vectorstores) if vectorstores else []
    wiki_tool = tool_creator.create_wiki()
    tools = tool_creator.make_tools(wiki_tool, retrievers)

    generator = Generator()
    prompt = generator.create_prompt(tools)
    executor = generator.create_agent(prompt, memory, tools)

    return executor