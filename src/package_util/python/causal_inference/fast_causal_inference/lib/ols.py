class Ols:
    def __init__(self, str):
        if str.find("(Intercept)") != -1:
            self.use_bias = 1
        else:
            self.use_bias = 0

        head = str[str.find("Pr(>|t|)") + 9 :]
        pos = head.find("Residual")
        self.is_ols = True
        if pos == -1:
            self.is_ols = False
        tail = head[pos:]
        head = head[: pos - 4]

        self.estimate = []
        self.std_err = []
        self.t_values = []
        self.pr = []

        for item in head.split("\n"):
            if item.replace(" ", "") == "":
                continue
            raw = item.split(" ")
            raw = [x.strip() for x in raw if x.strip()]
            pos = 0

            self.estimate.append(round(float(raw[1 + pos]), 6))
            self.std_err.append(round(float(raw[2 + pos]), 6))
            self.t_values.append(round(float(raw[3 + pos]), 6))
            self.pr.append(round(float(raw[4 + pos]), 6))

        if self.is_ols == False:
            return

        tail = tail.replace("\n", "")
        tail = tail.replace("Residual standard error: ", "")
        tail = tail.replace("on ", "")
        tail = tail.replace(" degrees of freedomMultiple R-squared:", "")
        tail = tail.replace(", Adjusted R-squared:", "")
        tail = tail.replace("F-statistic:", "")
        tail = tail.replace(" and", "")
        tail = tail.replace(" DF,  p-value:", "")
        tail = tail.replace("degrees of freedom  Multiple R-squared: ", "")
        tail_list = tail.split(" ")
        tail_list = [item for item in tail_list if item is not None and item != ""]

        self.standard_error = round(float(tail_list[0]), 6)
        self.df = int(tail_list[1])
        self.multiple_r_squared = round(float(tail_list[2]), 6)
        self.adjusted_r_squared = round(float(tail_list[3]), 6)
        self.f_statistic = round(float(tail_list[4]), 6)
        self.k = int(tail_list[5])
        self.f_df = int(tail_list[6])
        self.p_value = round(float(tail_list[7]), 6)

    def __str__(self):
        result = "Call:\nlm(formula = y ~"
        arg_num = len(self.estimate) - self.use_bias
        for i in range(arg_num):
            result += " + x" + str(i + 1)
        result += (
            ")\n\nCoefficients:\n\t\tEstimate    Std. Error\tt value\t    Pr(>|t|)\n"
        )

        ljust_len = 12

        if self.use_bias == 1:
            result += (
                "(Intercept)\t"
                + str(self.estimate[0]).ljust(ljust_len)
                + str(self.std_err[0]).ljust(ljust_len)
                + str(self.t_values[0]).ljust(ljust_len)
                + str(self.pr[0]).ljust(ljust_len)
                + "\n"
            )

        for i in range(arg_num):
            result += (
                "x"
                + str(i + self.use_bias)
                + "\t\t"
                + str(self.estimate[i + self.use_bias]).ljust(ljust_len)
                + str(self.std_err[i + self.use_bias]).ljust(ljust_len)
                + str(self.t_values[i + self.use_bias]).ljust(ljust_len)
                + str(self.pr[i + self.use_bias]).ljust(ljust_len)
                + "\n"
            )

        if self.is_ols:
            result += (
                "\nResidual standard error: "
                + str(self.standard_error)
                + " on "
                + str(self.df)
                + " degrees of freedom\nMultiple R-squared: "
                + str(self.multiple_r_squared)
                + ", Adjusted R-squared: "
                + str(self.adjusted_r_squared)
                + "\nF-statistic: "
                + str(self.f_statistic)
                + " on "
                + str(self.k)
                + " and "
                + str(self.f_df)
                + " DF,  p-value: "
                + str(self.p_value)
            )

        return result

    def get_estimate(self):
        return self.estimate

    def get_stderr(self):
        return self.std_err

    def get_t_values(self):
        return self.t_values

    def get_pr(self):
        return self.pr

    def get_dml_summary(self):
        result = "\t\tCoefficient Results\n"
        arg_num = len(self.estimate) - self.use_bias
        result += "\t\tEstimate    Std. Error\tt value\t    Pr(>|t|)\n"
        ljust_len = 12

        for i in range(arg_num - 1):
            result += (
                "x"
                + str(i)
                + "\t\t"
                + str(self.estimate[i]).ljust(ljust_len)
                + str(self.std_err[i]).ljust(ljust_len)
                + str(self.t_values[i]).ljust(ljust_len)
                + str(self.pr[i]).ljust(ljust_len)
                + "\n"
            )

        result += "\n\t\tCATE Intercept Results\n"
        result += "\t\tEstimate    Std. Error\tt value\t    Pr(>|t|)\n"
        i = arg_num - 1
        result += (
            "cate_intercept"
            + "\t"
            + str(self.estimate[i]).ljust(ljust_len)
            + str(self.std_err[i]).ljust(ljust_len)
            + str(self.t_values[i]).ljust(ljust_len)
            + str(self.pr[i]).ljust(ljust_len)
            + "\n"
        )
        return result
