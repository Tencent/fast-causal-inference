__doc__ = """
因果推断Oteam 
提供常见因果分析的工具
pip install -U fast-causal-inference
"""

from .all_in_sql import *

PROJECT_CONF = dict()


def set_tenant(tenant_id, secret_key):
    global PROJECT_CONF
    from .common.rainbow import RainbowConfCenter
    if not tenant_id or not secret_key:
        raise Exception("please input tenant_id and secret_key")
    PROJECT_CONF = RainbowConfCenter(tenant_id=tenant_id, secret_key=secret_key).get_conf()


def set_config(conf_path):
    global PROJECT_CONF
    import yaml
    from yaml.loader import SafeLoader
    if not conf_path:
        raise Exception("please input tenant_conf")
    with open(conf_path, 'r') as f:
        PROJECT_CONF = yaml.load(f, Loader=SafeLoader)


import logging.config
import os
logging.config.fileConfig(os.path.abspath(__file__).replace("__init__.py", "conf/fast_causal_inference_logging.conf"))
logger = logging.getLogger('my_custom')

# logger.setLevel("INFO")
