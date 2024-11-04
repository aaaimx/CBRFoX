import cbr_fox
from factory.cbr_fox_builder import cbr_fox_builder
from custom_distance.cci_distance import cci_distance
import numpy as np

# Load the saved data
data = np.load("variables.npz")

# Retrieve each variable
training_windows = data['training_windows']
forecasted_window = data['forecasted_window']
target_training_windows = data['target_training_windows']
windowsLen = data['windowsLen'].item()  # Extract single value from array
componentsLen = data['componentsLen'].item()
windowLen = data['windowLen'].item()
prediction = data['prediction']

techniques = [
    cbr_fox.cbr_fox(metric=cci_distance,kwargs={"punishedSumFactor":.5}),
    cbr_fox.cbr_fox(metric=cci_distance,kwargs={"punishedSumFactor":.7})
]
p = cbr_fox_builder(techniques)
p.fit(training_windows = training_windows,target_training_windows = target_training_windows.reshape(-1,1), forecasted_window = forecasted_window)
p.predict(prediction = prediction,num_cases=5)
p.plot_correlation()