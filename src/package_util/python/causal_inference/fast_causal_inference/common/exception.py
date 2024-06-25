def handle_exception(worker):
    from fast_causal_inference.common import get_context

    worker_exception = worker.exception()
    if worker_exception:
        get_context().logger.exception(worker_exception)
