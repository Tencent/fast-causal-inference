from ..all_in_sql import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class LiftGainCurveResult:
    def __init__(self, data):
        self.result = pd.DataFrame(data, columns=['ratio', 'lift', 'gain', 'ate', 'ramdom_gain'])

    def __str__(self):
        return str(self.result)

    def summary(self):
        print(self.result)

    def get_result(self):
        return self.result


def get_lift_gain(ITE, Y, T, table, discrete_treatment=False, K=1000):
    sql = "select lift(" + str(ITE) + "," + str(Y) + "," + str(T) + "," + str(K) + "," + str(
        discrete_treatment).lower() + ") from " + str(table) + ' limit 100000'
    print(sql)
    sql_instance = create_sql_instance()
    result = sql_instance.sql(sql)
    if discrete_treatment == False:
        result = result[['ratio', 'lift', 'gain', 'ate', 'ramdom_gain']]
    return LiftGainCurveResult(result)

def hte_plot(results, labels=[]):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 4.8))
    if len(labels) == 0:
        labels = [f'model_{i + 1}' for i in range(len(results))]

    for i in range(len(results)):
        result = results[i].get_result()
        result = result.replace('nan', np.nan)
        result = result.astype(float)
        result = result.dropna()
        ax1.plot(result['ratio'], result['lift'], label=labels[i])
        ax2.plot([0] + list(result['ratio']), [0] + list(result['gain']), label=labels[i])
    ax1.plot(result['ratio'], result['ate'], label='Random Model')
    ax2.plot([0] + list(result['ratio']), [0] + list(result['ramdom_gain']), label='Random Model')
    ax1.set_title('Cumulative Lift Curve')
    ax1.legend()
    ax2.set_title('Cumulative Gain Curve')
    ax2.legend()
    fig.suptitle('Lift and Gain Curves')
    plt.show()



