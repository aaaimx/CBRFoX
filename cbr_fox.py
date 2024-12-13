import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sktime.distances import distance
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from custom_distance import sktime_interface
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# TODO Revisar si es conveniente agregar como atributo de clase a correlation_per_windows para facilitar el acceso en los métodos
# o si por tema de memoria sería adecuado solo almacenar smoothed_correlation

class cbr_fox:
    def __init__(self, metric: str or callable = "dtw", smoothness_factor: float = .2, kwargs: dict = {}):
        # Variables for setting

        self.metric = metric
        self.smoothness_factor = smoothness_factor
        self.kwargs = kwargs
        # Variables for results
        # self.outputComponentsLen = len(outputNames)
        self.smoothed_correlation = None
        self.analysisReport = None
        self.analysisReport_combined = None
        self.best_windows_index = list()
        self.worst_windows_index = list()
        self.bestMAE = list()
        self.worstMAE = list()
        # Private variables for easy access by private methods
        self.correlation_per_window = None
        self.input_data_dictionary = None
        self.records_array = None
        self.records_array_combined = None
        self.dtype = [('index', 'i4'),
                      ('window', 'O'),
                      ('target_window', 'O'),
                      ('correlation', 'f8'),
                      ('MAE', 'f8')]
        # PRIVATE METHODS. ALL THESE METHODS ARE USED INTERNALLY FOR PROCESSING AND ANALYSIS

    def _preprocess_input_data(self, training_windows, target_training_windows, forecasted_window):
        # gather some basic data from passed in variables
        input_data_dictionary = dict()
        input_data_dictionary['training_windows'] = training_windows
        input_data_dictionary['target_training_windows'] = target_training_windows
        input_data_dictionary['forecasted_window'] = forecasted_window
        input_data_dictionary['components_len'] = training_windows.shape[2]
        input_data_dictionary['window_len'] = training_windows.shape[1]
        input_data_dictionary['windows_len'] = len(training_windows)

        return input_data_dictionary

    def _smoothe_correlation(self):
        return lowess(self.__correlation_per_window, np.arange(len(self.__correlation_per_window)),
                      self.smoothness_factor)[:, 1]

    def _identify_valleys_peaks_indexes(self):
        return signal.argrelextrema(self.smoothed_correlation, np.less)[0], \
            signal.argrelextrema(self.smoothed_correlation, np.greater)[0]

    # TODO Analizar si conviene hacer que retorne los valores y luego asigne, con único fin de seguir el estándar
    def _retreive_concave_convex_segments(self, windows_len):
        self.concaveSegments = np.split(
            np.transpose(np.array((np.arange(windows_len), self.smoothed_correlation))),
            self.valley_index)
        self.convexSegments = np.split(
            np.transpose(np.array((np.arange(windows_len), self.smoothed_correlation))),
            self.peak_index)

    # TODO Analizar si conviene hacer que retorne los valores y luego asigne, con único fin de seguir el estándar
    def _retreive_original_indexes(self):
        for split in tqdm(self.concaveSegments, desc="Segmentos cóncavos"):
            self.best_windows_index.append(int(split[np.where(split == max(split[:, 1]))[0][0], 0]))
        for split in tqdm(self.convexSegments, desc="Segmentos convexos"):
            self.worst_windows_index.append(int(split[np.where(split == min(split[:, 1]))[0][0], 0]))

    def calculate_analysis(self, indexes, input_data_dictionary):
        return np.array([(index,
                                        input_data_dictionary["training_windows"][index],
                                        input_data_dictionary["target_training_windows"][index],
                                        self.correlation_per_window[index],
                                        mean_absolute_error(input_data_dictionary["target_training_windows"][index],
                                                            input_data_dictionary["prediction"].reshape(-1, 1)))
                                       for index in indexes], dtype=self.dtype)


    # def calculate_analysis_combined(self, input_data_dictionary):
    #    return np.array([(-index,
    #        np.mean(input_data_dictionary["training_windows"][
    #                dic[:input_data_dictionary["num_cases"]]], axis=0),
    #        np.mean(input_data_dictionary["target_training_windows"][
    #                dic[:input_data_dictionary["num_cases"]]], axis=0),
    #        np.mean(self.correlation_per_window[
    #                dic[:input_data_dictionary["num_cases"]]]),
    #        mean_absolute_error(
    #            np.mean(input_data_dictionary["target_training_windows"][
    #                    dic[:input_data_dictionary["num_cases"]]], axis=0),
    #            input_data_dictionary["prediction"].reshape(-1, 1)
    #        )
    #       ) for index, dic in enumerate([self.best_windows_index,self.worst_windows_index])], dtype=self.dtype)

    def calculate_analysis_combined(self, input_data_dictionary, mode):

        def weighted_average(values, weights):
            weights = np.array(weights)
            weights = weights[:, np.newaxis, np.newaxis]
            return np.sum(values * weights, axis=0) / np.sum(weights)
        results = []
        for index, indices in enumerate([self.best_windows_index, self.worst_windows_index]):
            selected_cases = indices[:input_data_dictionary["num_cases"]]

            # Promedio simple
            simple_average = np.mean(
                input_data_dictionary["training_windows"][selected_cases],
                axis=0
            )
            # Promedio ponderado
            weighted_average_result = weighted_average(
                input_data_dictionary["training_windows"][selected_cases],
                self.correlation_per_window[selected_cases]
            )

            target_average = np.mean(
                input_data_dictionary["target_training_windows"][selected_cases],
                axis=0
            )
            correlation_mean = np.mean(self.correlation_per_window[selected_cases])
            mae = mean_absolute_error(
                np.mean(input_data_dictionary["target_training_windows"][selected_cases], axis=0),
                input_data_dictionary["prediction"].reshape(-1, 1)
            )

            if mode == "weighted":
                results.append((-index, weighted_average_result, target_average, correlation_mean, mae))
            else:
                results.append((-index, simple_average, target_average, correlation_mean, mae))

        return np.array(results, dtype=self.dtype)

    # TODO Analizar si este método puede ser el único que permita realizar asignaciones de variable internamente
    def _compute_statistics(self, input_data_dictionary, mode):

        # self.bestDic = {index: self.__correlation_per_window[index] for index in self.best_windows_index}
        #
        # self.worstDic = {index: self.__correlation_per_window[index] for index in self.worst_windows_index}
        #
        # self.bestDic = sorted(self.bestDic.items(), reverse=True, key=lambda x: x[1])
        #
        # self.worstDic = sorted(self.worstDic.items(), key=lambda x: x[1])
        #
        # self.bestDic = self.bestDic[0:input_data_dictionary['num_cases']]
        # self.worstDic = self.worstDic[0:input_data_dictionary['num_cases']]
        #
        # print("Calculando MAE para cada ventana")
        #
        # for tupla in self.bestDic:
        #     self.bestMAE.append(
        #         mean_absolute_error(input_data_dictionary["target_training_windows"][tupla[0]],
        #                             input_data_dictionary["prediction"].reshape(-1, 1)))
        #
        # for tupla in self.worstDic:
        #     self.worstMAE.append(
        #         mean_absolute_error(input_data_dictionary["target_training_windows"][tupla[0]],
        #                             input_data_dictionary["prediction"].reshape(-1, 1)))

        self.records_array_combined = self.calculate_analysis_combined(input_data_dictionary, mode)

        self.records_array = self.calculate_analysis(self.best_windows_index + self.worst_windows_index,
                                                     input_data_dictionary)

        # Sorting the array
        self.records_array = np.sort(self.records_array, order="correlation")[::-1]

        # Selecting just the number of elements according to num_cases variable
        # The conditional is to avoid duplicity in case records_arrays's shape is not greater than the selected num_cases
        if (self.records_array.shape[0] > (input_data_dictionary["num_cases"]*2)):
            self.records_array = np.concatenate((self.records_array[:input_data_dictionary["num_cases"]], self.records_array[
                                                                                                 -input_data_dictionary[
                                                                                                     "num_cases"]:]))

        print("Generando reporte de análisis")
        self.analysisReport = pd.DataFrame(data=pd.DataFrame.from_records(self.records_array))
        self.analysisReport_combined = pd.DataFrame(data=pd.DataFrame.from_records(self.records_array_combined))

    def _compute_cbr_analysis(self, input_data_dictionary):
        logging.info("Suavizando Correlación")
        self.smoothed_correlation = self._smoothe_correlation()
        logging.info("Extrayendo crestas y valles")
        self.valley_index, self.peak_index = self._identify_valleys_peaks_indexes()
        logging.info("Recuperando segmentos convexos y cóncavos")
        self._retreive_concave_convex_segments(input_data_dictionary['windows_len'])
        logging.info("Recuperando índices originales de correlación")
        self._retreive_original_indexes()

    def _compute_correlation(self, input_data_dictionary):

        # Implementing interface architecture to reduce tight coupling.
        correlation_per_window = sktime_interface.compute_distance_interface(input_data_dictionary, self.metric,
                                                                             self.kwargs)
        correlation_per_window = np.sum(correlation_per_window, axis=1)
        correlation_per_window = ((correlation_per_window - min(correlation_per_window)) /
                                  (max(correlation_per_window) - min(correlation_per_window)))
        self.correlation_per_window = correlation_per_window
        return correlation_per_window

    # PUBLIC METHODS. ALL THESE METHODS ARE PROVIDED FOR THE USER

    def fit(self, training_windows: np.ndarray, target_training_windows: np.ndarray, forecasted_window: np.ndarray):

        logging.info("Analizando conjunto de datos")
        self.input_data_dictionary = self._preprocess_input_data(training_windows, target_training_windows,
                                                                         forecasted_window)
        logging.info("Calculando Correlación")
        self.__correlation_per_window = self._compute_correlation(self.input_data_dictionary)
        logging.info("Computando análisis de CBR")
        self._compute_cbr_analysis(self.input_data_dictionary)
        logging.info("Análisis finalizado")


    def predict(self,prediction, num_cases: int, mode = "simple"):
        self.input_data_dictionary['prediction'] = prediction
        self.input_data_dictionary['num_cases'] = num_cases
        #aqui
        self._compute_statistics(self.input_data_dictionary, mode)

    # Method to print a chart or graphic based on results stored in variables. These methods are not strictly necessary
    #   for underlying functionality
    def visualize_correlation_per_window(self, plt_oject):
        pass

    def visualize_pyplot(self, **kwargs):
        import matplotlib.pyplot as plt
        figs_axes = []
        num_plots = self.input_data_dictionary["target_training_windows"].shape[1]

        # Un plot por cada componente
        for i in range(num_plots):
            fig, ax = plt.subplots()

            # Plot forecasted window and prediction
            ax.plot(
                np.arange(self.input_data_dictionary["window_len"]),
                self.input_data_dictionary["forecasted_window"][:, i],
                '--dk',
                label=kwargs.get("forecast_label", "Forecasted Window")
            )
            ax.scatter(
                self.input_data_dictionary["window_len"],
                self.input_data_dictionary["prediction"][i],
                marker='d',
                c='#000000',
                label=kwargs.get("prediction_label", "Prediction")
            )

            # Plot best windows
            for index in self.best_windows_index:
                plot_args = [
                    np.arange(self.input_data_dictionary["window_len"]),
                    self.input_data_dictionary["training_windows"][index, :, i]
                ]
                if "fmt" in kwargs:
                    plot_args.append(kwargs["fmt"])
                ax.plot(
                    *plot_args,
                    **kwargs.get("plot_params", {}),
                    label=kwargs.get("windows_label", f"Window {index}")
                )
                ax.scatter(
                    self.input_data_dictionary["window_len"],
                    self.input_data_dictionary["target_training_windows"][index, i],
                    **kwargs.get("scatter_params", {})
                )

            ax.set_xlim(kwargs.get("xlim"))
            ax.set_ylim(kwargs.get("ylim"))
            ax.set_xticks(np.arange(self.input_data_dictionary["window_len"]))
            plt.xticks(rotation=kwargs.get("xtick_rotation", 0), ha=kwargs.get("xtick_ha", 'right'))
            ax.set_title(kwargs.get("title", f"Plot {i + 1}"))
            ax.set_xlabel(kwargs.get("xlabel", "Axis X"))
            ax.set_ylabel(kwargs.get("ylabel", "Axis Y"))

            if kwargs.get("legend", True):
                ax.legend()

            figs_axes.append((fig, ax))
            fig.show()
        return figs_axes

    def get_analysis_report(self):
        return self.analysisReport

    def get_analysis_report_combined(self):
        return self.analysisReport_combined
