class Ttest:
    def __init__(self, str):
        last_line = str.splitlines()[-1]
        str_list = last_line.split(' ')
        str_list = list(filter(lambda x: x != '', str_list))
        self.estimate = round(float(str_list[0]), 6)
        self.stderr = round(float(str_list[1]), 6)
        self.t_statistic = round(float(str_list[2]), 6)
        self.p_value = round(float(str_list[3]), 6)
        self.lower = round(float(str_list[4]), 6)
        self.upper = round(float(str_list[5]), 6)

    def __str__(self):
        str = "estimate\tstderr\t\tt-statistic\tp-value\t\tlower\t\tupper\n"
        str += "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % (self.estimate, self.stderr, self.t_statistic, self.p_value, self.lower, self.upper)
        return str

    def get_estimate(self):
        return self.estimate

    def get_stderr(self):
        return self.stderr

    def get_t_statistic(self):
        return self.t_statistic

    def get_p_value(self):
        return self.p_value

    def get_lower(self):
        return self.lower

    def get_upper(self):
        return self.upper
