from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import logging
from src.exception import CustomException
import sys

class DataIngestor:
    """
    Class to load and process documents.
    """

    def load_documents(self, pdf_files: list) -> list:
        """
        Loads documents from PDF files.

        Args:
            pdf_files (list): List of PDF file paths.

        Returns:
            list: List of loaded documents.

        Raises:
            CustomException: If an error occurs while loading documents.
        """
        try:
            docs = []
            for filepath in pdf_files:
                loader = PyPDFLoader(filepath)
                docs.append(loader.load())
            logging.info("Loaded the PDF documents")
            return docs
        except Exception as e:
            logging.info("Error occurred while loading the PDF documents")
            raise CustomException(e, sys)

    def make_chunks(self, docs: list) -> list:
        """
        Splits documents into chunks.

        Args:
            docs (list): List of loaded documents.

        Returns:
            list: List of document chunks.

        Raises:
            CustomException: If an error occurs while chunking or retrieving page content.
        """
        try:
            documents = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            for doc in docs:
                splitted_docs = text_splitter.split_documents(doc)
                documents.append(splitted_docs)
            logging.info("Chunks created")

            contents = []
            for document in documents:
                page_content = [pages.page_content for pages in document]
                contents.append(page_content)
            logging.info("Page content retrieved")
            return contents

        except Exception as e:
            logging.error("Error in chunking")
            raise CustomException(e, sys)