import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Generate dataset:
def data_generator(datasets, timestep):
    # Arguments:
    #   datasets: scaled_data
    #   timestep: number of previous day that today depend on
    # Output:
    #   input_data: numpy array; shape: (size(datsets) - timestep, timestep) 
    #   target_data: numpy array; shape: (size(datsets) - timestep)
    dataset_inp = []
    dataset_oup = []
    for day in range(timestep, len(datasets)):
        subsample = []
        for prev_day in range(day-timestep, day):
            subsample.append(datasets[prev_day])
        dataset_inp.append(subsample)
        dataset_oup.append(datasets[day])
    return (np.array(dataset_inp), np.array(dataset_oup))

# plot dataset
def plot_data(datasets, token):
    # Arguments:
    #   datasets: pdFrame
    # ticker = datasets['ticker'][0]
    plt.plot(datasets, color='blue')
    plt.xlabel("Date")
    plt.ylabel("Close price")
    plt.title(f"{token}")
    plt.show()