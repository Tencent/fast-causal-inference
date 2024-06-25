import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fast_causal_inference.util import SqlGateWayConn


# Define a class to store the results of Lift Gain Curve
class LiftGainCurveResult:
    def __init__(self, data):
        # Initialize the result as a DataFrame with specified column names
        self.result = pd.DataFrame(
            data, columns=["ratio", "lift", "gain", "ate", "ramdom_gain"]
        )

    def __str__(self):
        # Return the string representation of the result
        return str(self.result)

    def summary(self):
        # Print the summary of the result
        print(self.result)

    def get_result(self):
        # Return the result
        return self.result


# Function to calculate lift gain
def get_lift_gain(ITE, Y, T, table, normalize=True, K=1000, discrete_treatment=True):
    """
    Inputs:
    ITE: The Individual Treatment Effect
    Y: The outcome variable
    T: The treatment variable
    table: The table name in the database
    normalize: Whether to normalize the result, default is True
    K: The number of bins for discretization, default is 1000
    discrete_treatment: Whether the treatment is discrete, default is True

    Outputs:
    A LiftGainCurveResult object containing the result of the lift gain calculation
    """

    # Construct the SQL query
    sql = (
        "select lift("
        + str(ITE)
        + ","
        + str(Y)
        + ","
        + str(T)
        + ","
        + str(K)
        + ","
        + str(discrete_treatment).lower()
        + ") from "
        + str(table)
        + " limit 100000"
    )
    # Create an SQL instance
    sql_instance = SqlGateWayConn.create_default_conn()
    # Execute the SQL query and get the result
    result = sql_instance.sql(sql)
    # Select specific columns from the result
    result = result[["ratio", "lift", "gain", "ate", "ramdom_gain"]]
    # Replace 'nan' with np.nan
    result = result.replace("nan", np.nan)
    # Convert the data type to float
    result = result.astype(float)
    # Drop rows with missing values
    result = result.dropna()
    # Normalize the result if required
    if normalize:
        result = result.div(np.abs(result.iloc[-1, :]), axis=1)
    # Calculate AUUC
    auuc = result["gain"].sum() / result.shape[0]
    print("auuc:", auuc)
    # Return the result as a LiftGainCurveResult object
    return LiftGainCurveResult(result)


# Function to plot HTE
def hte_plot(results, labels=[]):
    """
    Inputs:
    results: A list of LiftGainCurveResult objects to be plotted
    labels: A list of labels for the results, default is an empty list

    Outputs:
    None. This function will display a plot.
    """

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 4.8))
    # Generate labels if not provided
    if len(labels) == 0:
        labels = [f"model_{i + 1}" for i in range(len(results))]
    # Plot the results
    for i in range(len(results)):
        result = results[i].get_result()
        auuc = round(result["gain"].sum() / result.shape[0], 2)
        ax1.plot(result["ratio"], result["lift"], label=labels[i])
        ax2.plot(
            [0] + list(result["ratio"]),
            [0] + list(result["gain"]),
            label=labels[i] + f"(auuc:{auuc})",
        )
    # Plot the ATE and random gain
    ax1.plot(result["ratio"], result["ate"])
    ax2.plot([0] + list(result["ratio"]), [0] + list(result["ramdom_gain"]))
    # Set the titles and legends
    ax1.set_title("Cumulative Lift Curve")
    ax1.legend()
    ax2.set_title("Cumulative Gain Curve")
    ax2.legend()
    # Set the title for the figure
    fig.suptitle("Lift and Gain Curves")
    # Display the figure
    plt.show()
