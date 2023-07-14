from rainbow_sdk.rainbow_client import RainbowClient
import yaml
from yaml.loader import SafeLoader
import io

"""
七彩石配置中心
QPS较小，无需使用文件缓存 + watch机制
"""
class RainbowConfCenter(object):
    GROUP = "ALL_IN_SQL"
    ENV_NAME = "Default"
    CONNECT_STR = "api.rainbow.oa.com:8080"

    def __init__(self, tenant_id, secret_key):
        if not tenant_id or not secret_key:
            raise Exception("please input tenant_id and secret_key")
        if "$" not in tenant_id:
            raise Exception("tenant_id value format is error")
        self.app_id = tenant_id.split("$")[0]
        self.user_id = tenant_id.split("$")[1]
        self.secret_key = secret_key
        self.init_param = {
            "connectStr": RainbowConfCenter.CONNECT_STR,
            "tokenConfig": {
                "app_id": self.app_id,
                "user_id": self.user_id,
                "secret_key": self.secret_key,
            },
        }

    def get_conf(self):
        rainbow_client = RainbowClient(self.init_param)
        conf = rainbow_client.get_configs_v3(group=RainbowConfCenter.GROUP,
                                             env_name=RainbowConfCenter.ENV_NAME)["data"]["conf.yaml"]
        conf = conf.decode("utf-8").replace("\\n", "\n").strip()
        conf_dict = yaml.load(io.StringIO(conf).read(), Loader=SafeLoader)
        return conf_dict

