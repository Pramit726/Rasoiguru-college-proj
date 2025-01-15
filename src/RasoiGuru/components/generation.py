from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from src.logger import logging
from src.exception import CustomException
import sys

class Generator:
    """
    Class to handle user interactions and generate responses.
    """

    def __init__(self):
        """
        Initializes the Generator.
        """
        # load_dotenv()
        # os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        # print(f"API Key: {os.getenv('GROQ_API_KEY')}")
        self.llm = ChatGroq(model="mixtral-8x7b-32768")

    def create_prompt(self, tools: list) -> PromptTemplate:
        """
        Creates the prompt template for the language model.

        Args:
            tools (list): List of available search tools.

        Returns:
            PromptTemplate: The prompt template for the language model.

        Raises:
            CustomException: If an error occurs while creating the prompt.
        """
        try:
            system_instruction = """
            You are a helpful cooking assistant named Rasoiguru.
            Greet the user.
            Answer the following questions as best you can in terms of a passionate and helpful professional cooking assistant.
            """

            format = """
            Use the following format:

            Use the chat history which will be provided to you for understanding the context of the most recent conversation in case user query is not clearly defined.
            Question: the input question you must answer.
            Thought: you should always think about what to do.
            Action: the action to take, should be one of the provided tools.
            Action Input: the input to the action.
            Observation: the result of the action.
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer.
            Final Answer: the final answer to the original input question.

            Remember to answer as a compassionate professional cooking assistant when giving your final answer.
            """

            prefix = f""" You have access to the following tools:
            Tools:
            {tools}
            Instruction:
            {system_instruction}.
            """

            suffix = """Begin! Now answer the question
            {intermediate_steps}
            Chat history:
            {chat_history}
            Question: {input}
            {agent_scratchpad}
            ## In case the user query is not about food cooking, grocery shopping, and history of food,
            then reply I do not know the answer to your question.
            ## You need to always provide the answer after writing Final Answer: \
            """

            prompt = PromptTemplate(
                input_variables=["input", "chat_history", "intermediate_steps", "agent_scratchpad"],
                template= prefix + format + suffix
            )

            logging.info("Prompt created successfully")

            return prompt

        except Exception as e:
            logging.info("Error occurred while creating the prompt")
            raise CustomException(e, sys)

    def create_agent(self, prompt: PromptTemplate, memory: ConversationBufferWindowMemory, tools: list) -> AgentExecutor:
        """
        Creates an agent and agent executor.

        Args:
            prompt (PromptTemplate): The prompt template for the language model.
            memory (ConversationBufferWindowMemory): Conversation memory.
            tools (list): List of available search tools.

        Returns:
            AgentExecutor: The agent executor.

        Raises:
            CustomException: If an error occurs while creating the agent or agent executor.
        """
        try:
            agent = create_tool_calling_agent(self.llm, tools=tools, prompt=prompt)
            logging.info("Agent created successfully")

            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
            logging.info("Agent executor created successfully")
            return agent_executor
        except Exception as e:
            logging.info("Error occurred while creating the agent executor")
            raise CustomException(e, sys)