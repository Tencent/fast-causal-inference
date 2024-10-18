import os
from fast_causal_inference.common import get_context


def build_spark_session(
    group_id,
    gaia_id,
    cmk=None,
    driver_cores=4,
    driver_memory="8g",
    executor_cores=2,
    executor_memory="10g",
    **kwargs,
):
    raise NotImplementedError("This function is not implemented yet.")
