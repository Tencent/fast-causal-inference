import os
import pandas


def get_user():
    if str(os.environ.get("JUPYTERHUB_USER")) != "None":
        return str(os.environ.get("JUPYTERHUB_USER"))
    elif str(os.environ.get("USER")) != "None":
        return str(os.environ.get("USER"))
    elif str(os.environ.get("CURRENT_USER")) != "None":
        return str(os.environ.get("CURRENT_USER"))
    else:
        return str('default')


def output_auto_boxing(res):
    lines = res.split('\n')
    results = []
    for i in range(len(lines)):
        if len(lines[i]) != 0:
            columns = lines[i].split('\t')
            result_line = []
            for j in range(len(columns)):
                if columns[j].isdigit():
                    columns[j] = int(columns[j])
                else:
                    try:
                        columns[j] = float(columns[j])
                    except Exception:
                        pass
                result_line.append(columns[j])
            results.append(result_line)
    return results


def output_dataframe(res):
    return pandas.DataFrame(list(eval(res)))

def to_pandas(res):
    i = 1
    data = list()
    for line in res.splitlines():
        if i == 1:
            columns = list(filter(lambda x: x != '', line.split(' ')))
            i += 1
        else:
            data.append(list(filter(lambda x: x != '', line.split(' '))))
    return pandas.DataFrame(data, columns=columns)
