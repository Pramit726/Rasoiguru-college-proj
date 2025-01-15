import sys
from src.logger import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Formats and returns a detailed error message.

    Args:
        error (Exception): The exception instance.
        error_detail (sys): The sys module, used to extract exception information.

    Returns:
        str: A formatted string containing the filename, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in Python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    """
    Custom exception class that provides detailed error messages.

    Attributes:
        error_message (str): Detailed error message.
    """

    def __init__(self, error_message: str, error_detail: sys) -> None:
        """
        Initializes the CustomException with a detailed error message.

        Args:
            error_message (str): The error message.
            error_detail (sys): The sys module, used to extract exception information.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        """
        Returns the detailed error message when the exception is converted to a string.

        Returns:
            str: The detailed error message.
        """
        return self.error_message




        