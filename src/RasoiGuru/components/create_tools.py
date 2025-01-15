from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool
from langchain.tools.retriever import create_retriever_tool
from src.logger import logging
from src.exception import CustomException
import sys

class ToolCreator:
    """
    Class to create search tools.
    """

    def create_retriever(self, vectorstores: list) -> list:
        """
        Creates retrievers from vectorstores.

        Args:
            vectorstores (list): List of PineconeVectorStore objects.

        Returns:
            list: List of retriever objects.

        Raises:
            CustomException: If an error occurs while creating retrievers.
        """
        try:
            retrievers = []
            for vectorstore in vectorstores:
                retriever = vectorstore.as_retriever()
                retrievers.append(retriever)
            logging.info("Retrievers created successfully")
            return retrievers
        except Exception as e:
            logging.error("Error creating retrievers")
            raise CustomException(e, sys)

    def create_wiki(self) -> Tool:
        """
        Creates the Wikipedia search tool.

        Returns:
            Tool: The Wikipedia search tool.

        Raises:
            CustomException: If an error occurs while creating the Wikipedia tool.
        """
        try:
            wiki_tool = WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(
                    top_k_results=1,
                    load_all_available_meta=False,
                    doc_content_chars_max=500
                )
            )
            wiki_tool = Tool(
                name='Wikipedia',
                description='look up things in wikipedia for knowing about food recipes, cooking instructions and their history',
                func=wiki_tool.invoke
            )
            logging.info("Wikipedia tool created successfully")
            return wiki_tool
        except Exception as e:
            logging.error("Error creating Wikipedia tool")
            raise CustomException(e, sys)

    def make_tools(self, wiki_tool: Tool, retrievers: list) -> list:
        """
        Creates all search tools.

        Args:
            wiki_tool (Tool): The Wikipedia search tool.
            retrievers (list): List of retriever objects.

        Returns:
            list: List of all search tools.

        Raises:
            CustomException: If an error occurs while creating tools.
        """
        try:
            tools_name = [
                'BHM-401T_pdf_search'
            ]
            tools_desc = [
                "Indian food cooking and heritage related information use this tool"
            ]

            tools = []
            for name, desc, retv in zip(tools_name, tools_desc, retrievers):
                pdf_tool = create_retriever_tool(retv, name, desc, document_prompt="Search the query")
                tools.append(pdf_tool)

            tools.append(wiki_tool)
            logging.info("Tools created successfully")
            return tools
        except Exception as e:
            logging.error("Error creating tools")
            raise CustomException(e, sys)