from src.core import cbr_fox
from src.builder import cbr_fox_builder
from src.custom_distance.cci_distance import cci_distance
import numpy as np

# Load the saved data
data = np.load("Walmart_Sales.npz")

# Retrieve each variable
training_windows = data['training_windows']
forecasted_window = data['forecasted_window']
target_training_windows = data['target_training_windows']
windowsLen = data['windowsLen'].item()  # Extract single value from array
componentsLen = data['componentsLen'].item()
windowLen = data['windowLen'].item()
prediction = data['prediction']

techniques = [
    cbr_fox.cbr_fox(metric=cci_distance, kwargs={"punishedSumFactor": 0.5})
    #cbr_fox.cbr_fox(metric="edr"),
    #cbr_fox.cbr_fox(metric="dtw"),
    #cbr_fox.cbr_fox(metric="twe")
]
p = cbr_fox_builder(techniques)
p.fit(training_windows = training_windows,target_training_windows = target_training_windows, forecasted_window = forecasted_window)
p.predict(prediction = prediction,num_cases=3)
# p.plot_correlation()

p.visualize_pyplot(
    fmt = '-.',
    legend = False,
    scatter_params={"s": 25},
    xtick_rotation=50,
    title="Walmart Sales",
    xlabel="Day",
    ylabel="Metric Value"
)