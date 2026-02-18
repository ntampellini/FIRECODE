import logging

import pytest


@pytest.fixture(autouse=True)
def restore_logging_handlers():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    yield
    root_logger.handlers = original_handlers
