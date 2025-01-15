from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import time
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from src.exception import CustomException
from src.logger import logging
import sys

class IndexManager:
    """
    Class to manage Pinecone index creation and document insertion.
    """

    def __init__(self, index_name: str, cloud: str = "aws", region: str = "us-east-1"):
        """
        Initializes the IndexManager.

        Args:
            index_name (str): Name of the Pinecone index.
            cloud (str, optional): Cloud provider (e.g., "aws", "gcp"). Defaults to "aws".
            region (str, optional): Region for the Pinecone index. Defaults to "us-east-1".
        """
        load_dotenv()
        os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.pc = Pinecone()

    def create_index(self) -> Pinecone.Index:
        """
        Creates a Pinecone index if it doesn't exist.

        Returns:
            Pinecone.Index: The created or retrieved Pinecone index.

        Raises:
            CustomException: If an error occurs during index creation.
        """
        try:
            if self.index_name not in self.pc.list_indexes().names():
                spec = ServerlessSpec(cloud=self.cloud, region=self.region)
                self.pc.create_index(
                    self.index_name,
                    dimension=4096,
                    metric='cosine',
                    spec=spec
                )

            index = self.pc.Index(self.index_name)
            time.sleep(1)
            index.describe_index_stats()
            logging.info("Index created successfully")
            return index

        except Exception as e:
            logging.info("Error creating index")
            raise CustomException(e, sys)

    def insert_documents(self, pdf_files: list, contents: list) -> list[PineconeVectorStore]:
        """
        Inserts documents into the Pinecone index.

        Args:
            pdf_files (list): List of PDF file paths.
            contents (list): List of document contents.

        Returns:
            list[PineconeVectorStore]: List of PineconeVectorStore objects.

        Raises:
            CustomException: If an error occurs during vector insertion.
        """
        try:
            ns = ["ns" + path.stem for path in pdf_files]
            embedding_model = CohereEmbeddings()
            logging.info("Embedding model loaded")
            vectorstores = []

            if self.pc.Index(self.index_name).describe_index_stats()['total_vector_count'] == 0:
                for namespace, content in zip(ns, contents):
                    vectorstore = PineconeVectorStore.from_texts(
                        texts=content,
                        index_name=self.index_name,
                        embedding=embedding_model,
                        namespace=namespace
                    )
                    vectorstores.append(vectorstore)
                logging.info("Inserted the vectors")
            else:
                for namespace in ns:
                    vectorstore = PineconeVectorStore.from_existing_index(self.index_name, embedding_model, namespace=namespace)
                    vectorstores.append(vectorstore)
                logging.info("Received the vector stores from an existing index")

            return vectorstores

        except Exception as e:
            logging.error("Error inserting vectors")
            raise CustomException(e, sys)