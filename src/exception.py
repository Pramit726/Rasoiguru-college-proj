from src.logger import logging
import sys

# Function to format error message with details
def error_message_detail(error, error_detail: sys) -> str:
    """
    Formats the error message with details.

    Args:
        error: The error object.
        error_detail: The error details from the sys module.

    Returns:
        A formatted error message string containing file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    """
    A custom exception class for handling errors.

    This class extends the base Exception class and provides a more detailed error message
    including file name and line number.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException object.

        Args:
            error_message: The original error message.
            error_detail: The error details from the sys module.
        """
        super().__init__(error_message)  # Call the base class constructor
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """
        Returns the formatted error message as a string.
        """
        return self.error_message



        