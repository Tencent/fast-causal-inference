import json
import time
from urllib import request
from urllib.error import HTTPError
from urllib.request import Request
from .tdw_tauth_authentication import TdwTauthAuthentication
from .. import logger
from .. import PROJECT_CONF

class IdexUtils(object):
    IDEX_CLUSTERS = PROJECT_CONF["idex"]["clusters"]
    IDEX_TASKS = PROJECT_CONF["idex"]["tasks"]

    def __init__(self, user, cmk):
        self.tauth = TdwTauthAuthentication(user, cmk, "idex-openapi")

    def get_headers(self):
        headers = {
            'Version': '2',
            'Authentication': self.tauth.getAuthentication()['authentication'],
            'Content-Type': 'application/json'
        }
        return headers

    def run_sql(self, sql):
        logger.debug('获取集群列表')
        clusters_request = Request(IdexUtils.IDEX_CLUSTERS, headers=self.get_headers())
        try:
            with request.urlopen(clusters_request) as response:
                clusters_data = json.load(response)
        except HTTPError as exception:
            message = json.load(exception)['message']
            logger.debug(message)
            raise
        logger.debug(clusters_data)

        logger.debug('获取单个集群')
        cluster_request = Request(clusters_data[0], headers=self.get_headers())
        try:
            with request.urlopen(cluster_request) as response:
                cluster_data = json.load(response)
        except HTTPError as exception:
            message = json.load(exception)['message']
            logger.debug(message)
            raise
        logger.debug(cluster_data)

        logger.debug('获取资源池列表')
        pools_request = Request(cluster_data['pools_url'], headers=self.get_headers())
        try:
            with request.urlopen(pools_request) as response:
                pools_data = json.load(response)
        except HTTPError as exception:
            message = json.load(exception)['message']
            logger.debug(message)
            raise
        logger.debug(pools_data)

        logger.debug('获取单个资源池')
        pool_request = Request(pools_data[0], headers=self.get_headers())
        try:
            with request.urlopen(pool_request) as response:
                pool_data = json.load(response)
        except HTTPError as exception:
            message = json.load(exception)['message']
            logger.debug(message)
            raise
        logger.debug(pool_data)

        logger.debug('运行单个任务')
        request_data = json.dumps({
            'statements': sql,
            'cluster_id': 'tl',
            'group_id': pool_data['group_id'],
            'database': 'wxg_weixin_experiment',
            'gaia_id': pool_data['gaia_id']
        }).encode()
        logger.debug("request_data=" + str(request_data))
        tasks_request = Request(IdexUtils.IDEX_TASKS, request_data, self.get_headers())
        try:
            with request.urlopen(tasks_request) as response:
                tasks_data = json.load(response)
        except HTTPError as exception:
            message = json.load(exception)['message']
            logger.debug(message)
            raise
        logger.debug(tasks_data)

        # 必须等待5秒
        time.sleep(5)

        logger.debug('获取单个任务')
        task_request = Request(tasks_data['task_url'], headers=self.get_headers())
        try:
            with request.urlopen(task_request) as response:
                task_data = json.load(response)
        except HTTPError as exception:
            message = json.load(exception)['message']
            logger.debug(message)
            raise
        logger.debug("task_data=" + str(task_data))

        while True:
            logger.debug('获取语句列表')
            try:
                statements_request = Request(task_data['statements_url'], headers=self.get_headers())
                with request.urlopen(statements_request) as response:
                    statements_data = json.load(response)
            except HTTPError as exception:
                message = json.load(exception)['message']
                logger.debug(message)
                raise
            time.sleep(3)
            logger.debug("statements_data=" + str(statements_data) + " " + str(len(statements_data)) + " "
                         +  str(sql.strip().strip(";").split(";").__len__()))
            if len(statements_data) == sql.strip().strip(";").split(";").__len__():
                break

        res = list()
        urls = set()
        while True:
            logger.debug('获取单个任务')
            task_request = Request(tasks_data['task_url'], headers=self.get_headers())
            try:
                with request.urlopen(task_request) as response:
                    task_data = json.load(response)
            except HTTPError as exception:
                message = json.load(exception)['message']
                logger.debug(message)
                raise
            logger.debug(task_data)
            for statement_url in statements_data:
                if statement_url not in urls:
                    logger.debug('获取单个语句')
                    statement_request = Request(statement_url, headers=self.get_headers())
                    try:
                        with request.urlopen(statement_request) as response:
                            statement_data = json.load(response)
                    except HTTPError as exception:
                        message = json.load(exception)['message']
                        logger.debug(message)
                        raise
                    logger.debug(statement_data)

                    if statement_data['state'] == 'success':
                        logger.debug('获取结果')
                        result_request = Request(statement_data['result_url'], headers=self.get_headers())
                        try:
                            with request.urlopen(result_request) as response:
                                result_data = response.read()
                        except HTTPError as exception:
                            message = json.load(exception)['message']
                            logger.debug(message)
                            raise
                        res_str = str(result_data)
                        logger.debug(res_str)
                        res.append(res_str)
                        start = res_str.find("location:")
                        end = res_str.find(",", start)
                        location = res_str[start + len("location:"):end]
                        logger.debug(location)
                        urls.add(statement_url)

            if task_data['state'] in ('success', 'failure', 'abortion'):
                break
        return res

    def get_table_meta(self, db_name, table_name):
        res = self.run_sql("""
                                    set `supersql.bypass.forceAll`=false; 
                                    use %s;
                                    desc %s;
                                    desc extended %s;
                                    """ % (db_name, table_name, table_name))
        desc = res[2]
        col_names = list()
        col_types = list()
        partitions = list()
        for row in desc.replace("\\xef\\xbb\\xbf", "").replace("\\x01", ",").split("\\n")[1:]:
            field = row.split(",")
            col_name = field[0]
            col_names.append(col_name)
            col_type = field[1]
            col_types.append(col_type)
        logger.debug(col_names)
        logger.debug(col_types)
        desc_extended = res[3]
        if "priPartition" in desc_extended and "subPartition" in desc_extended:
            is_thive = True
            if "priPartition:null" not in desc_extended:
                is_partition_table = True
            else:
                is_partition_table = False
        else:
            is_thive = False
            is_partition_table = False
        if "ParquetInputFormat" in desc_extended:
            table_format = "Parquet"
        elif "OrcInputFormat" in desc_extended:
            table_format = "ORC"
        elif "TextInputFormat" in desc_extended:
            table_format = "TEXT"
        else:
            raise Exception("table_format is not support, desc_extended=" + str(desc_extended))
        location = desc_extended[desc_extended.find("location:") + len("location:"):
                                 desc_extended.find(",", desc_extended.find("location"))]
        logger.debug("is_thive=" + str(is_thive) + ",is_partition_table=" + str(is_partition_table) + ",location="
                     + str(location))
        if is_partition_table:
            res_partitions = self.run_sql("""
                                    set `supersql.bypass.forceAll`=false; 
                                    use %s;
                                    show partitions %s;
                                    """ % (db_name, table_name))
            show_partitions = res_partitions[2]
            for row in show_partitions.replace("\\xef\\xbb\\xbf", "").split("\\n")[2:]:
                partition = row.strip("'")
                partitions.append(partition)
            logger.debug(partitions)
        return col_names, col_types, partitions, location, table_format


if __name__ == '__main__':
    pass

