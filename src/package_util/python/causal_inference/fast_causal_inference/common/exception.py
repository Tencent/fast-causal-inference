import logging

logger = logging.getLogger(__name__)

def handle_exception(worker):
    worker_exception = worker.exception()
    if worker_exception:
        logger.exception(worker_exception)
