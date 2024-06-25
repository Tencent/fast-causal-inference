import numpy as np
import scipy.stats as stats

from fast_causal_inference.util import SqlGateWayConn


def IPWestimator(table, Y, T, P, B=500):
    """
    Estimate the Average Treatment Effect (ATE) using Inverse Probability of Treatment Weighting (IPTW).

    Args:
    table: str, the name of the input data table.
    Y: str, the column name of the outcome variable.
    T: str, the column name of the treatment variable.
    P: str, the column name of the propensity score.
    B: int, the number of bootstrap samples, default is 500.

    Returns:
    dict, containing the following key-value pairs:
        'ATE': Average Treatment Effect.
        'stddev': Standard deviation.
        'p_value': p-value.
        '95% confidence_interval': 95% confidence interval.
    """
    # Create SQL instance
    sql_instance = SqlGateWayConn.create_sql_instance()

    # Get the number of rows in the table
    n = int(sql_instance.sql(f"select count(*) as cnt from {table}")["cnt"][0])

    # Execute SQL query to calculate IPTW estimates
    res = (
        sql_instance.sql(
            f"""WITH (
      SELECT DistributedNodeRowNumber(1)(0)
      FROM {table}
    ) AS pa
    SELECT
      BootStrapMulti('sum:1;sum:1;sum:1;sum:1',  {n}, {B}, pa)(
      {Y}*{T}/({P}+0.01), {T}/({P}+0.01), {Y}*(1-{T})/(1-{P}+0.01), (1-{T})/(1-{P}+0.01)) as res
    FROM 
    {table}
    ;
    """
        )["res"][0]
        .replace("]", "")
        .replace(" ", "")
        .split("[")
    )

    # Process the query results
    res = [i.split(",") for i in res if i != ""]
    res = np.array([[float(j) for j in i if j != ""] for i in res])

    # Calculate IPTW estimates
    result = res[0, :] / res[1, :] - res[2, :] / res[3, :]
    ATE = np.mean(result)

    # Calculate standard deviation
    std = np.std(result)

    # Calculate t-value
    t_value = ATE / std

    # Calculate p-value
    p_value = (1 - stats.t.cdf(abs(t_value), n - 1)) * 2

    # Calculate 95% confidence interval
    confidence_interval = [ATE - 1.96 * std, ATE + 1.96 * std]

    # Return results
    return {
        "ATE": ATE,
        "stddev": std,
        "p_value": p_value,
        "95% confidence_interval": confidence_interval,
    }


def ATEestimator(table, Y, T, B=500):
    """
    Estimate the Average Treatment Effect (ATE) using a simple difference in means approach.

    Args:
    table: str, the name of the input data table.
    Y: str, the column name of the outcome variable.
    T: str, the column name of the treatment variable.
    B: int, the number of bootstrap samples, default is 500.

    Returns:
    dict, containing the following key-value pairs:
        'ATE': Average Treatment Effect.
        'stddev': Standard deviation.
        'p_value': p-value.
        '95% confidence_interval': 95% confidence interval.
    """
    # Create SQL instance
    sql_instance = SqlGateWayConn.create_sql_instance()

    # Get the number of rows in the table
    n = int(sql_instance.sql(f"select count(*) as cnt from {table}")["cnt"][0])

    # Execute SQL query to compute ATE estimator using a simple difference in means approach

    res = (
        sql_instance.sql(
            f"""WITH (
      SELECT DistributedNodeRowNumber(1)(0)
      FROM {table}
    ) AS pa
    SELECT
      BootStrapMulti('sum:1;sum:1;sum:1;sum:1',  {n}, {B}, pa)(
      {Y}*{T},{T},{Y}*(1-{T}),(1-{T})) as res
    FROM 
    {table}
    ;
    """
        )["res"][0]
        .replace("]", "")
        .replace(" ", "")
        .split("[")
    )

    # Process the query results
    res = [i.split(",") for i in res if i != ""]
    res = np.array([[float(j) for j in i if j != ""] for i in res])

    # Calculate IPTW estimates
    result = res[0, :] / res[1, :] - res[2, :] / res[3, :]

    # Compute the ATE
    ATE = np.mean(result)

    # Compute standard deviation
    std = np.std(result)

    # Compute t-value
    t_value = ATE / std

    # Compute p-value
    p_value = (1 - stats.t.cdf(abs(t_value), n - 1)) * 2

    # Compute 95% confidence interval
    confidence_interval = [ATE - 1.96 * std, ATE + 1.96 * std]

    # Return the results
    return {
        "ATE": ATE,
        "stddev": std,
        "p_value": p_value,
        "95% confidence_interval": confidence_interval,
    }
