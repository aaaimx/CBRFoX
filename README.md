# CBR-FoX: Case-Based Reasoning for Time Series Prediction Explanations

CBR-FoX is a Python library designed to provide case-based reasoning explanations for time series prediction models. This approach enhances the transparency and understanding of machine learning models used with sequential data.

## Features

- CBR-FoX approach implementation.
- Adaptable to various types of time series.
- Compatible with common machine learning models.
- Generates comprehensible explanations.

## Installation

Clone this repository and install its dependencies:
```bash
git clone https://github.com/jerryperezperez/CBR-FoX.git
cd CBR-FoX
pip install -r requirements.txt
```

## Usage

Follow these steps to use CBR-FoX in your projects:

1. **Retreive model's information:**
   Obtain the inputs and outputs generated by your AI model.


2. **Create CBRfox instances:**
   ```python
   cbr_instances = CBRfoxInstances(model_outputs)
   ```

3. **Initialize Builder** 
   ```Python 
    builder = CBRfoxBuilder(cbr_instances)
    ```
4. **Train the instance:**
   ```Python 
    builder.fit(train_windows, train_targets, target_to_analyze, window_to_predict)
    ```
5. **Obtain explanations:**
   ```Python 
    builder.predict(prediction = prediction,num_cases=5)
    ```
6. **Use graph visualization methods:**
    ```Python 
    builder.visualize_pyplot(
        fmt = '--d',
        scatter_params={"s": 50},
        xtick_rotation=50,
        title="nombre",
        xlabel="x",
        ylabel="y"
    )
    ```
### Library Usage Diagram

The following diagram illustrates the typical workflow when using the CBR-FoX library. From retrieving inputs and outputs from the AI model to generating visual explanations, each step is designed to facilitate the interpretation and explanation of time series-based predictions.

![Library basic usage diagram](https://github.com/jerryperezperez/CBR-FoX/blob/develop/library_basic_usage_diagram.svg)

### Library file relation diagram

The following diagram shows the classes involved in the basic functionality of the library. The`cci_distance` file is used when creating an instance that employs the eponymous technique implemented in this script.

![Library file relation diagram](https://github.com/jerryperezperez/CBR-FoX/blob/develop/file_relation_diagram.svg)
