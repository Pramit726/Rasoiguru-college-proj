from src.logger import logging
from src.exception import CustomException
import sys
import time
from pathlib import Path
from pinecone import Pinecone


def extract_answer(result: str) -> str:
    """Extracts the final answer from the provided response string.

    This function searches for a string marker "Final Answer:" within the response
    and returns the text following it as the final answer. If the marker is not found,
    an empty string is returned.

    Args:
        result: The response string from the language model.

    Returns:
        The extracted final answer as a string.

    Raises:
        CustomException: If an error occurs during processing.
    """

    try:
        result = str(result)
        start_marker = "Final Answer:"
        start_index = result.find(start_marker)

        if start_index != -1:
            result = result[start_index + len(start_marker):].strip()
        else:
            result = ""

        logging.info("Successfully extracted final answer")
        return result

    except Exception as e:
        logging.error("Error in extracting final answer")
        raise CustomException(e, sys)


def vector_exist(index_name: str, pc: Pinecone) -> bool:
    """Checks if any vectors exist in the specified Pinecone index.

    This function attempts to describe the index statistics using the Pinecone
    client. If the total vector count is zero, it indicates that no vectors exist
    in the index. Otherwise, it assumes vectors are present.

    Args:
        index_name: The name of the Pinecone index to check.
        pc: A Pinecone client instance.

    Returns:
        True if vectors exist in the index, False otherwise.

    Raises:
        CustomException: If an error occurs during communication with Pinecone.
    """

    try:
        index = pc.Index(index_name)
        # Wait a moment for connection
        time.sleep(1)
        if index.describe_index_stats()['total_vector_count'] == 0:
            logging.info("Vectors do not exist")
            return False
        else:
            logging.info("Vector already exists")
            return True

    except Exception as e:
        logging.info("Error checking vector existence")
        raise CustomException(e, sys)


def get_paths(data_dir: Path = Path("data")) -> list[Path]:
    """Retrieves the paths of all PDF files within a specified directory.

    This function takes an optional `data_dir` argument specifying the directory
    containing the PDF files. It iterates through the files in the directory and
    returns a list of paths to all files with the `.pdf` extension.

    Args:
        data_dir (Path, optional): The directory containing the PDF files.
            Defaults to "data" in the parent directory of the current script.

    Returns:
        A list of Path objects representing the paths to the PDF files.

    Raises:
        CustomException: If an error occurs while accessing the directory or files.
    """

    try:
        # Get the current working directory
        current_path = Path(__file__)

        # Get the parent directory of the current working directory
        parent_path = current_path.parent.parent

        # Construct the path to the data directory (default or provided)
        data_path = parent_path / data_dir

        pdf_files = []
        for file in data_path.iterdir():
            if file.is_file() and file.name.endswith('.pdf'):
                pdf_files.append(file)

        logging.info("Got the PDF file paths")
        return pdf_files

    except Exception as e:
        logging.info("Error occurred while getting the PDF file paths")
        raise CustomException(e, sys)