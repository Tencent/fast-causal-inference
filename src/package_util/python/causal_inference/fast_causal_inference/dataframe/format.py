import pandas
import warnings 
import numpy as np

def format_db_result(result_list, pd=None):
    if pd is None:
        pd = pandas.DataFrame(result_list)
    pd = pd.replace('�', np.nan)

    ttest_width = 1
    for row in range(len(pd)):
        head = []
        max_count = 1
        is_exception = False
        for col in pd.columns:
            first_item = pd[col][row]
            if isinstance(first_item, str) and first_item.find("Data are essentially") != -1:
                is_exception = True
            res = format_each_item(first_item)
            if res is None:
                head.append(col)
            else:
                for i in range(len(res[0])):
                    head.append(res[0][i])
                max_count = max(max_count, len(res[1][0]))
                ttest_width = max(ttest_width, len(res[1][0][0]))
        if is_exception == False:
            break

    if ttest_width == 5:
        ttest_width = 14

    content = []
    pd_result = []
    for row in range(len(pd)):
        content = []
        for col in pd.columns:
            first_item = pd[col][row]

            res = format_each_item(first_item)
            if res is None:
                if isinstance(first_item, str) and first_item.find("Data are essentially") != -1:
                    print('Data are essentially constant (standard error is zero), making t-statistic undefined. Check for variability in your data at {} row.'.format(row + 1))
                    content.append([[''] * ttest_width])
                else:
                    content.append([[first_item]])
            else:
                for i in range(len(res[1])):
                    content.append(res[1][i])
        for i in range(max_count):
            new_row = []
            for j in content:
                items = j[0]
                if i < len(j):
                    items = j[i]
                for item in items:
                    new_row.append(item)
            pd_result.append(new_row)

        max_length = max(len(nested) for nested in pd_result)
        for nested in pd_result:
            nested.extend([""] * (max_length - len(nested)))

    pd = pandas.DataFrame(pd_result, columns=head)
    # if pd's column is numeric, convert it to numeric, otherwise, keep it as string.
    return pd

def convert_to_numeric(pd):
    for col in pd.columns:
        try:
            pd[col] = pd[col].apply(pandas.to_numeric)
        except:
            pass
    return pd

def format_each_item(content):
    if isinstance(content, list):
        content = content[0]
    elif isinstance(content, str):
        if content.startswith("["):
            content = content[1:-1]
    if not isinstance(content, (str, list, tuple)) or "p-value" not in content or "Coefficient" in content:
        return None

    is_xexpt_format = "recommend_samples" in content
    content = content.split("\n")
    content = [i.split() for i in content if i]
    content = [i for i in content if i != [","]]
    if is_xexpt_format == False:
        head = content[0]
        content = [content[i] for i in range(len(content)) if i % 2 == 1]
    else:
        head = content[0] + content[3]
        content = [content[i] for i in range(len(content)) if i % 5 != 0 and i % 5 != 3]
        for i in range(0, len(content), 3):
            content[i + 1] += content[i + 2]
        content = [x for i, x in enumerate(content) if i % 3 != 2]

    return head, [content]



## Internal helper function for formatting output
def format_result(pd_df, target_col):
    if pd_df.shape[0]==1:
        return float(pd_df[target_col][0])
    else:
        return np.array(pd_df[target_col].astype(float))
        

## Internal helper fucntion for handling utest pandas dataframe 
def process_utest_result_pd(utest_result_pd):
    c_names = utest_result_pd.columns 
    assert ('__result__' in c_names ) or (np.sum(['mannWhitneyUTest' in c for c in c_names]) > 0)
    if '__result__' in c_names:
        result_col = '__result__'
    else: 
        result_col = [c for c in c_names if 'mannWhitneyUTest' in c][0]
    
    
    ustat_array = np.array(utest_result_pd[result_col].apply(lambda x: eval(x)[0]))
    upval_array = np.array(utest_result_pd[result_col].apply(lambda x: eval(x)[1]))
    
    final_result = utest_result_pd.copy()
    final_result = final_result.drop(result_col, axis = 1)
    final_result['u-statistic'] = ustat_array 
    final_result['p-value'] = upval_array
    return final_result


class testResult:
    """
    This class stores the results of statistical tests for easy manipulation of the results.

    Attributes: 
        p_value (float): the p-value of the corresponding test if exists. 
        statistic (float): the (observed) test statistic of the corresponding test if exists.
        conf_int (tuple): (lower, upper) confidence interval of the corresponding test if exists.
        estimate (float): the estiamted difference in means. 
        stderr (float): the standard error of the mean (difference), used as denominator of t-statistic. 
        mean0 (float): the sample mean of the control data if exists. 
        mean1 (float): the sample mean of the treat data if exists. 

    """
    
    def __init__(self, result_ch_df):
        self.result_ch_df = result_ch_df
        self.result_pd = result_ch_df.toPandas()
        ## Hard-code for utest result 
        if ("__result__" in self.result_pd) or (np.sum(['mannWhitneyUTest' in c for c in self.result_pd.columns]) > 0):
            self.result_pd = process_utest_result_pd(self.result_pd)
        self.test_result_attributes = list(self.result_pd.columns)
        
    @property 
    def p_value(self):
        if 'p-value' in self.test_result_attributes:
            return format_result(self.result_pd, 'p-value') 
        else:
            warnings.warn("The test did not return a p-value.")
            return np.nan
        
    @property 
    def statistic(self):
        ## Check the column that corresponds to the statistics 
        stat_col = [c for c in self.test_result_attributes if 'statistic' in c ]
        assert((len(stat_col) == 1 ) or (len(stat_col) == 0))
        if len(stat_col) == 0:
            warnings.warn("The test did not return a test statistic.")
            return np.nan 
        
        
        stat_col_name = stat_col[0]
        return format_result(self.result_pd, stat_col_name) 
    
    @property 
    def conf_int(self):
        if ('lower' in self.test_result_attributes) and ('upper' in self.test_result_attributes):
            return np.array([format_result(self.result_pd, "lower"),format_result(self.result_pd, "upper")]).transpose()
    
        else:
            
            warnings.warn("The test did not return a confidence interval.")
            return (np.nan, np.nan)
        
        
    @property 
    def estimate(self):
        if 'estimate' in self.test_result_attributes:
            return format_result(self.result_pd, 'estimate') 
        else:
            warnings.warn("The test did not return an estiamte.")
            return np.nan

    @property 
    def stderr(self):
        if 'stderr' in self.test_result_attributes:
            return format_result(self.result_pd, 'stderr') 
        else:
            warnings.warn("The test did not return a stderr.")
    @property 
    def mean0(self):
        if 'mean0' in self.test_result_attributes:
            return format_result(self.result_pd, 'mean0') 
        else:
            warnings.warn("The test did not return a mean for control data.")
            return np.nan
    @property 
    def mean1(self):
        if 'mean1' in self.test_result_attributes:
            return format_result(self.result_pd, 'mean1') 
        else:
            warnings.warn("The test did not return a mean for treat datas.")
            return np.nan
    def toPandas(self):
        return self.result_pd
    
    def __str__(self):
        ## 这里的__str__ 保留了 pandas dataframe自带的str function
        return self.result_ch_df.__str__()
    
    def __repr__(self):
        ## 这里的__repr__ 保留了 pandas dataframe自带的str function
        return self.result_ch_df.__repr__()
    
    def show(self):
        return self.result_ch_df.show()

