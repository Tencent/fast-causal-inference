from .all_in_sql import *
from .common.rainbow import RainbowConfCenter

__doc__ = """
因果推断Oteam 
提供常见因果分析的工具
pip install -i https://xx:xxx@mirrors.tencent.com/pypi/simple -U fast-causal-inference
"""

_global_tenant_id = ""
_global_secret_key = ""
RAINBOW_CONF = dict()

def set_tenant(tenant_id, secret_key):
    global _global_tenant_id, _global_secret_key, RAINBOW_CONF
    _global_tenant_id = tenant_id
    _global_secret_key = secret_key
    RAINBOW_CONF = RainbowConfCenter(tenant_id=_global_tenant_id, secret_key=_global_secret_key).get_conf()

import logging.config
import os
logging.config.fileConfig(os.path.abspath(__file__).replace("__init__.py", "conf/fast_causal_inference_logging.conf"))
logger = logging.getLogger('my_custom')
# logger.setLevel("INFO")
