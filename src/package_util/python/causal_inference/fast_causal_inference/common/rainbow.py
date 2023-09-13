from rainbow_sdk.rainbow_client import RainbowClient
import yaml
from yaml.loader import SafeLoader
import io
import os

"""
七彩石配置中心
QPS较小，无需使用文件缓存 + watch机制
"""

class RainbowConfCenter(object):

    def __init__(self, tenant_id, secret_key):
        if not os.environ.get("RAINBOW_URL") and not os.environ.get("RAINBOW_GROUP") and not os.environ.get("RAINBOW_ENV"):
            self.url = str(os.environ.get("RAINBOW_URL"))
            self.group = str(os.environ.get("RAINBOW_GROUP"))
            self.env = str(os.environ.get("RAINBOW_ENV"))
        else:
            raise Exception("RAINBOW environment is empty, please check")
        if not tenant_id or not secret_key:
            raise Exception("please input tenant_id and secret_key")
        if "$" not in tenant_id:
            raise Exception("tenant_id value format is error")
        self.app_id = tenant_id.split("$")[0]
        self.user_id = tenant_id.split("$")[1]
        self.secret_key = secret_key
        self.init_param = {
            "connectStr": self.url,
            "tokenConfig": {
                "app_id": self.app_id,
                "user_id": self.user_id,
                "secret_key": self.secret_key,
            },
        }

    def get_conf(self):
        rainbow_client = RainbowClient(self.init_param)
        conf = rainbow_client.get_configs_v3(group=self.group, env_name=self.env)["data"]["conf.yaml"]
        conf = conf.decode("utf-8").replace("\\n", "\n").strip()
        conf_dict = yaml.load(io.StringIO(conf).read(), Loader=SafeLoader)
        return conf_dict
