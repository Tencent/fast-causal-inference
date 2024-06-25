import pandas


def output_auto_boxing(res):
    lines = res.split("\n")
    results = []
    for i in range(len(lines)):
        if len(lines[i]) != 0:
            columns = lines[i].split("\t")
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
    if "error message" in res:
        return res
    else:
        return pandas.DataFrame(list(eval(res)))


def to_pandas(res):
    data = list()
    columns = None
    for line in res:
        i = 1
        for inner_line in line.splitlines():
            if not inner_line:
                continue
            if not columns:
                columns = list(filter(lambda x: x != "", inner_line.split(" ")))
            if i >= 2:
                data.append(list(filter(lambda x: x != "", inner_line.split(" "))))
            i += 1
    return pandas.DataFrame(data, columns=columns)
