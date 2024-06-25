import warnings

import numpy as np

from fast_causal_inference.util import SqlGateWayConn

warnings.filterwarnings("ignore")


def kaplan_meier(table, time_column="first_hit_ds", censor_col=None):
    """This function generates the survival probabilities by Kaplan Meier estimator.

    Parameters
    ----------
    table: str
        The name of the ClickHouse data table.

    time_column : str
        The name of the column that contain the time information. The data in the time column
        must be convertible to int type.

    censor_col : str, optional
        The name of the column that indicates the censoring status. 0 indicating that the failure
        time happens after the observed time. None by default - observations are not censored.

    Returns
    -------
    survival_probabilities: list
        a list of survival probabilities, whose index starts from t = 0.


    """

    sql_instance = SqlGateWayConn.create_default_conn()
    if censor_col is not None:
        df_for_km = sql_instance.sql(
            f"""select count(*) as n, {t} from {table} where {censor_col}=1 group by {time_column}"""
        )
    else:
        df_for_km = sql_instance.sql(
            f"""select count(*) as n, {t} from {table} group by {time_column}"""
        )
    all_ds = df_for_km[time_column].unique().astype("int")
    sorted_ds = sorted(all_ds)
    mapping_dict = {}
    for i in range(len(sorted_ds)):
        mapping_dict[str(sorted_ds[i])] = i
    df_for_km["t_hopefullynotintheoldtable"] = df_for_km[time_column].map(mapping_dict)

    times = list(range(1, len(sorted_ds) + 1))
    deaths = df_for_km.sort_values("t_hopefullynotintheoldtable")["n"].astype(int)
    at_risk = (
        np.sum(df_for_km.sort_values("t_hopefullynotintheoldtable")["n"].astype(int))
        - np.append(
            [0],
            np.cumsum(
                df_for_km.sort_values("t_hopefullynotintheoldtable")["n"].astype(int)
            ),
        )
    )[: df_for_km.shape[0]]
    survival_prob = np.cumprod(1 - deaths / at_risk)
    return survival_prob
