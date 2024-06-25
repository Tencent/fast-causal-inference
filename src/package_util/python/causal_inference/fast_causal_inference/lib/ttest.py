from io import StringIO
import pandas


class Ttest:
    def __init__(self, ttest_result):
        self.ttest_result = ttest_result
        ttest_result_parts = self.ttest_result.split("\n\n")

        self.summary = pandas.read_csv(StringIO(ttest_result_parts[0]), sep="\s+")
        data_parts2 = ""
        start = False
        for cell in ttest_result_parts[1]:
            if cell == "[":
                start = True
            if cell == "]":
                start = False
            if start and cell == " ":
                data_parts2 += ""
            else:
                data_parts2 += cell
        self.result = pandas.read_csv(StringIO(data_parts2), sep="\s+")

    def __str__(self):
        return self.ttest_result
