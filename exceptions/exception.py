from loguru import logger

class CustomException(Exception):
    """Custom Exception to standardize error handling."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def handle_exception(e):
    """Handles exceptions and logs them."""
    if isinstance(e, CustomException):
        logger.error(f"Custom Exception: {e.message}")
    else:
        logger.error(f"An unexpected error occurred: {str(e)}")
