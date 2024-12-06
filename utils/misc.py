from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from utils import metrics
import logging 
import pickle

class TerminateNaN(Callback):
    """
    Callback that terminates training if NaN loss is encountered.
    """
    
    def __init__(self, monitor='loss'):
        """
        Initializes the TerminateNaN callback.

        :param monitor: The metric to monitor; defaults to 'loss'.
        """
        super().__init__()
        self.monitor = monitor


import numpy as np
import logging
import csv  # Assuming you need to import csv for file operations
from utils import metrics  # Assuming this is where `metrics` is defined
import numpy as np
import logging
import csv  # Assuming you need to import csv for file operations
from utils import metrics  # Replace with your actual metrics module import
import pickle

class EvaluationMetric:
    
    EPSILON = 1e-10

    def __init__(self, INCLUDE_LAGS, INCLUDE_SEASON_VARS, INCLUDE_WEATHER, TUNER = "Hyperband"):

        self.include_lags = INCLUDE_LAGS
        self.include_season_vars = INCLUDE_SEASON_VARS
        self.include_weather = INCLUDE_WEATHER
        self.tuner = TUNER

         # You can add any initialization code here if needed in the future

    def mape(self, actual, forecast):
        """
        Calculate Mean Absolute Percentage Error (MAPE) using NumPy for efficient computation.
        
        Parameters:
            actual (list or array): Actual values.
            forecast (list or array): Forecasted values.
            
        Returns:
            float: MAPE value, expressed as a percentage.
        """
        actual, forecast = np.array(actual), np.array(forecast)
        # Use np.where to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            percentage_errors = np.abs((actual - forecast) / actual)
            percentage_errors = np.where(actual == 0, 0, percentage_errors)

        # Calculate the mean of the percentage errors
        return np.mean(percentage_errors) 

    def wape(self, actual, forecast):
        """
        Calculate Weighted Absolute Percentage Error (WAPE).
        
        Parameters:
            actual (list or array): Actual values.
            forecast (list or array): Forecasted values.
            
        Returns:
            float: WAPE value.
        """
        actual, forecast = np.array(actual), np.array(forecast)
        absolute_errors = np.sum(np.abs(actual - forecast))
        total_actuals = np.sum(np.abs(actual))
        return absolute_errors / (total_actuals + self.EPSILON)

    def _error(self, actual: np.ndarray, predicted: np.ndarray):
        """ Simple error """
        return actual - predicted

    def _percentage_error(self, actual: np.ndarray, predicted: np.ndarray):
        """ Percentage error """
        return self._error(actual, predicted) / (actual + self.EPSILON)

    def mse(self, actual: np.ndarray, predicted: np.ndarray):
        """ Mean Squared Error """
        return np.mean(np.square(self._error(actual, predicted)))

    def rmse(self, actual: np.ndarray, predicted: np.ndarray):
        """ Root Mean Squared Error """
        return np.sqrt(self.mse(actual, predicted))

    def mae(self, actual: np.ndarray, predicted: np.ndarray):
        """ Mean Absolute Error """
        return np.mean(np.abs(self._error(actual, predicted)))

    def evaluate_and_log(self, y_test, predictions, time_period, results_dir, best_hps, duration, model='LSTM'):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

        # Adjust the directory names based on the inclusion of lags and season variables
        extra_info = f'_{self.tuner}'
        if not self.include_lags:
            extra_info += "_ex_lags"
        if not self.include_season_vars:
            extra_info += "_ex_season_dummies"
        if not self.include_weather:
            extra_info += "_ex_wheater"

        hyperparameter_file_path = f'{results_dir}/hyperparameters/{model}/{model}_best_hyperparameters_{time_period}{extra_info}.csv'
        metrics_file_path = f'{results_dir}/metrics/{model}/{model}_test_metrics_{time_period}{extra_info}.csv'

        if model.lower() == 'lstm':
            self.log_lstm_hyperparameters(logger, best_hps, hyperparameter_file_path)
        elif model.lower() == 'tcn':
            self.log_tcn_hyperparameters(logger, best_hps, hyperparameter_file_path)
        elif model.lower() == 'transformer':
            self.log_transformer_hyperparameters(logger, best_hps, hyperparameter_file_path)
        elif model.lower() == 'hybrid':
            self.log_hybrid_hyperparameters(logger, best_hps, hyperparameter_file_path)

        # Log test metrics
        test_metrics = metrics.evaluate_all(np.array(y_test), predictions)
        logger.info('Test scores:')
        logger.info(test_metrics)

        # Save test metrics to CSV
        with open(metrics_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['duration', duration])
            for key, value in test_metrics.items():
                writer.writerow([key, value])

        with open(f'{results_dir}/predictions/{model}/{model}_{time_period}_{self.tuner}_predictions_{extra_info}.pkl', 'wb') as f:
            pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)

    def log_lstm_hyperparameters(self, logger, best_hps, file_path):
        logger.info('Best Model Hyperparameters (LSTM):')
        logger.info(f'Units: {best_hps.get("units")}')
        logger.info(f'Dense units: {best_hps.get("dense_units")}')
        logger.info(f'Optimizer: {best_hps.get("optimizer")}')
        logger.info(f'Dropout 1: {best_hps.get("dropout_1")}')
        logger.info(f'Dropout 2: {best_hps.get("dropout_2")}')
        
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['units', best_hps.get('units')])
            writer.writerow(['dense_units', best_hps.get('dense_units')])
            writer.writerow(['optimizer', best_hps.get('optimizer')])
            writer.writerow(['dropout_1', best_hps.get('dropout_1')])
            writer.writerow(['dropout_2', best_hps.get('dropout_2')])

    def log_tcn_hyperparameters(self, logger, best_hps, file_path):
        logger.info('Best Model Hyperparameters (TCN):')
        logger.info(f'Kernel size: {best_hps.get("kernel_size")}')
        logger.info(f'Number of filters: {best_hps.get("nb_filters")}')
        logger.info(f'Number of stacks: {best_hps.get("nb_stacks")}')
        logger.info(f'Optimizer: {best_hps.get("optimizer")}')
        logger.info(f'Learning rate: {best_hps.get("lr_adam")}')
        logger.info(f'Dropout rate: {best_hps.get("dropout_rate")}')

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['kernel_size', best_hps.get('kernel_size')])
            writer.writerow(['nb_filters', best_hps.get('nb_filters')])
            writer.writerow(['nb_stacks', best_hps.get('nb_stacks')])
            writer.writerow(['optimizer', best_hps.get('optimizer')])
            writer.writerow(['lr_adam', best_hps.get('lr_adam')])
            writer.writerow(['dropout_rate', best_hps.get('dropout_rate')])

    def log_hybrid_hyperparameters(self, logger, best_hps, file_path):
        logger.info('Best Model Hyperparameters (Hybrid):')
        logger.info(f'Kernel size: {best_hps.get("kernel_size")}')
        logger.info(f'Number of filters: {best_hps.get("nb_filters")}')
        logger.info(f'Number of stacks: {best_hps.get("nb_stacks")}')
        logger.info(f'Optimizer: {best_hps.get("optimizer")}')
        logger.info(f'Learning rate: {best_hps.get("lr_adam")}')
        logger.info(f'Dropout rate: {best_hps.get("dropout_rate")}')
        logger.info(f'LSTM units: {best_hps.get("lstm_units")}')
        logger.info(f'Dense units: {best_hps.get("dense_units")}')

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['kernel_size', best_hps.get('kernel_size')])
            writer.writerow(['nb_filters', best_hps.get('nb_filters')])
            writer.writerow(['nb_stacks', best_hps.get('nb_stacks')])
            writer.writerow(['optimizer', best_hps.get('optimizer')])
            writer.writerow(['lr_adam', best_hps.get('lr_adam')])
            writer.writerow(['dropout_rate', best_hps.get('dropout_rate')])
            writer.writerow(['lstm_units', best_hps.get('lstm_units')])
            writer.writerow(['dense_units', best_hps.get('dense_units')])

    def log_transformer_hyperparameters(self, logger, best_hps, file_path):
        logger.info('Best Model Hyperparameters (Transformer):')
        logger.info(f'Head size: {best_hps.get("head_size")}')
        logger.info(f'Number of heads: {best_hps.get("num_heads")}')
        logger.info(f'Feed-forward dimension: {best_hps.get("ff_dim")}')
        logger.info(f'Number of Transformer blocks: {best_hps.get("num_transformer_blocks")}')
        logger.info(f'MLP units: {best_hps.get("mlp_units")}')
        logger.info(f'Dropout rate: {best_hps.get("dropout")}')
        logger.info(f'Optimizer: {best_hps.get("optimizer")}')
        logger.info(f'Learning rate: {best_hps.get("learning_rate")}')
        logger.info(f'Pooling type: {best_hps.get("pooling_type")}')

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['head_size', best_hps.get('head_size')])
            writer.writerow(['num_heads', best_hps.get('num_heads')])
            writer.writerow(['ff_dim', best_hps.get('ff_dim')])
            writer.writerow(['optimizer', best_hps.get('optimizer')])
            writer.writerow(['num_transformer_blocks', best_hps.get('num_transformer_blocks')])
            writer.writerow(['mlp_units', best_hps.get('mlp_units')])
            writer.writerow(['dropout_rate', best_hps.get('dropout')])
            writer.writerow(['learning_rate', best_hps.get('learning_rate')])
            writer.writerow(['pooling_type', best_hps.get('pooling_type')])


    def predict_future_values(self, datasets, scaler_y, final_model, n_hours_ahead=24):
        """
                Predict future values using a trained model iteratively for specified hours ahead.

                Parameters:
                - datasets: Dictionary containing the test feature sets (should include 'X_test' key).
                - scaler_y: Fitted scaler object to inverse transform the predictions.
                - final_model: The trained model to use for making predictions.
                - n_hours_ahead: Number of hours to predict ahead (default is 24).

                Returns:
                - predictions: Numpy array
            # def predict_future_values(self, datasets, scaler_y, final_model, n_hours_ahead=24):
            - predictions: Numpy array of predicted values.
        """
        # Get the number of test samples you want to predict
        n_test_samples = datasets['X_test'].shape[0]
        print(f"Total test samples: {n_test_samples}")

        # Initialize variables for predictions and current input
        predictions = []
        current_input = datasets['X_test'][:n_hours_ahead]  # Start with the first n_hours_ahead input

        # Iteratively make predictions for complete n_hours_ahead chunks
        while len(predictions) < n_test_samples:
            
            # Get the model's predicted price values for the current input
            predictions_scaled = final_model.predict(current_input)
            predicted_values = scaler_y.inverse_transform(predictions_scaled)
            
            # Append the predictions to the predictions list
            predictions.extend(predicted_values.flatten().tolist())
            print(f'Length of predictions: {len(predictions)}')

            # Prepare the input for the next iteration
            # Check if we can get another full n_hours_ahead
            if len(predictions) % n_hours_ahead == 0 and len(predictions) + n_hours_ahead <= n_test_samples:
                
                # Shift input: Update the current input to include data for the next prediction period
                start_index = len(predictions)  # current index for predictions
                current_input = datasets['X_test'][start_index:start_index + n_hours_ahead]
            else:
                break


        # Convert the predictions list to a numpy array for further processing or analysis
        predictions = np.array(predictions)

        return predictions

    def predict_full_sequence(self, model, X_test_scaled, scaler_y, input_steps=48, output_steps=24):
        """
        Predicts the full sequence for the test set using rolling windows.

        Parameters:
        - model: Trained model for making predictions.
        - X_test_scaled: Scaled test set features (NumPy array).
        - y_test_scaled: Scaled test set target values (NumPy array).
        - scaler_y: Fitted scaler for the target variable.
        - input_steps: Number of input steps (history length).
        - output_steps: Number of output steps (forecast horizon).

        Returns:
        - y_pred_full_rescaled: Predictions for the full test set, inverse-transformed to original scale.
        - y_test_full_rescaled: Actual test set values, inverse-transformed to original scale.
        """
        # Initialize variables
        n_test_samples = X_test_scaled.shape[0]
        y_pred_full = []

        # Start with the first input sequence
        current_input = X_test_scaled[:input_steps]  # The first input sequence

        # Full loop through the test data
        for i in range(0, n_test_samples - input_steps, output_steps):
            # Predict the next sequence
            y_pred_next = model.predict(current_input[np.newaxis, :, :], verbose=0)  # Add batch dimension
            y_pred_full.extend(y_pred_next[0])  # Append prediction (extend for time series continuity)

            # Update the current input sequence
            if i + input_steps + output_steps < n_test_samples:
                next_input = np.roll(current_input, shift=-output_steps, axis=0)  # Shift input
                next_input[-output_steps:] = X_test_scaled[input_steps + i:input_steps + i + output_steps]
                current_input = next_input

        # Handle the last remaining steps if any
        remaining_steps = n_test_samples - len(y_pred_full) - input_steps
        if remaining_steps > 0:
            # Use the last valid input sequence
            current_input = X_test_scaled[-(input_steps + output_steps):, :]  # Last valid sequence
            y_pred_next = model.predict(current_input[np.newaxis, :, :], verbose=0)
            y_pred_full.extend(y_pred_next[0][:remaining_steps])  # Append only the remaining steps

        # Convert predictions to a NumPy array
        y_pred_full = np.array(y_pred_full).reshape(-1, 1)  # Ensure it is a continuous time series

        # Inverse transform predictions
        predictions = scaler_y.inverse_transform(y_pred_full)

        return predictions

import matplotlib.pyplot as plt
import pandas as pd

class Plotting():
    def __init__(self, model, time_period, path_to_images, INCLUDE_LAGS, INCLUDE_SEASON_VARS, INCLUDE_WEATHER, TUNER = "Hyperband"):
        self.model = model
        self.time_period = time_period
        self.path_to_images = path_to_images
        self.include_lags = INCLUDE_LAGS
        self.include_season_vars = INCLUDE_SEASON_VARS
        self.include_weather = INCLUDE_WEATHER
        self.tuner = TUNER

                # Adjust the directory names based on the inclusion of lags and season variables
        self.extra_info = f'_{self.tuner}'
        if not self.include_lags:
            self.extra_info += "_ex_lags"
        if not self.include_season_vars:
            self.extra_info += "_ex_season_dummies"
        if not self.include_weather:
            self.extra_info += "_ex_wheater"

    def plot_training_history(self, history):
        """
        Plots the training history of a Keras model.

        :param history: The history object returned by the fit method of a Keras model.
        """
        plt.figure(figsize=(14, 5))

        # Summarize history for accuracy, if it exists
        if 'accuracy' in history.history:
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], color='navy')
            plt.plot(history.history['val_accuracy'], color='black')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')

        # Summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], color='navy')
        plt.plot(history.history['val_loss'], color='black')
        plt.title(f'{self.model} Model Loss for the period from {self.time_period}{self.extra_info}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()
        # Ensure the directory exists
        os.makedirs(f"{self.path_to_images}/{self.model}/", exist_ok=True)
        plt.savefig(f"{self.path_to_images}/{self.model}/{self.model}_model_loss_RandomSearch_{self.time_period}{self.extra_info}.png")
        plt.show()

    def plot_predictions(self, df, y, X_train, X_test, predictions):
        """
        Plots the actual vs predicted values.

        Parameters:
        - df: DataFrame containing the full dataset.
        - y: Actual target values.
        - X_train: Training features.
        - X_test: Test features.
        - predictions: Predictions to be plotted.
        """
        # Ensure a copy of the DataFrame to keep original intact
        df = df.copy()

        # Ensure `y` is a DataFrame
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, index=df.index, columns=["Day Ahead Spot Price"])
        try:
            # Calculate lengths of training and test datasets
            training_data_len = len(X_train)
            test_data_len = len(predictions)

            # Create subsets for train, valid, and test based on the data length
            train = y[:training_data_len].copy()
            valid = y[training_data_len:training_data_len + test_data_len].copy()
            test = y[-test_data_len:].copy()
            test['Predictions'] = predictions[:test_data_len]  # Add predictions to the test dataframe

        except Exception as e:
            print(f"An error occurred: {e}")
            # Adjust lengths if necessary
            training_data_len = len(X_train)
            test_data_len = len(X_test) - 1

            # Create subsets for train, valid, and test based on the data length
            train = y[:training_data_len].copy()
            valid = y[training_data_len:training_data_len + test_data_len].copy()
            test = y[training_data_len + test_data_len:].copy()
            test['Predictions'] = predictions  # Add predictions to the test dataframe

        # Assigning the index from the original dataframe to each subset
        train.index = df.index[:training_data_len]
        valid.index = df.index[training_data_len:training_data_len + test_data_len]
        test.index = df.index[-test_data_len:]

        # Combine into a single DataFrame for plotting
        combined_df = pd.concat([train, valid, test])
        combined_df = combined_df.iloc[:-50]  # Remove the last 50 rows to avoid overlap with predictions

        # Ensure the index is a DatetimeIndex and convert to date only for cleaner x-axis labels
        combined_df.index = pd.to_datetime(combined_df.index).date

        # Plotting setup
        plt.figure(figsize=(16, 8))
        plt.plot(combined_df.index[:training_data_len], combined_df['Day Ahead Spot Price'][:training_data_len], color='navy', label='Training Data')
        plt.plot(combined_df.index[training_data_len:training_data_len + test_data_len], combined_df['Day Ahead Spot Price'][training_data_len:training_data_len + test_data_len], color='black', label='Validation Data')
        plt.plot(combined_df.index[training_data_len + test_data_len:], combined_df['Day Ahead Spot Price'][training_data_len + test_data_len:], color='grey', label='Test Data')
        plt.plot(combined_df.index[training_data_len + test_data_len:], combined_df['Predictions'][training_data_len + test_data_len:], color='red', label='Predictions', alpha=0.7)

        # Final touches on plot
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Day Ahead Spot Price (DKK/MWh)')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # Ensure the directory exists
        os.makedirs(f"{self.path_to_images}/{self.model}/", exist_ok=True)
        plt.savefig(f"{self.path_to_images}/{self.model}/{self.model}_predictions_{self.time_period}{self.extra_info}.png", dpi=150, transparent=False)
        plt.show()

        # Plotting setup
        plt.figure(figsize=(16, 8))
        plt.plot(combined_df.index[training_data_len + test_data_len:], combined_df['Day Ahead Spot Price'][training_data_len + test_data_len:], color='grey', label='Test Data')
        plt.plot(combined_df.index[training_data_len + test_data_len:], combined_df['Predictions'][training_data_len + test_data_len:], color='red', label='Predictions', alpha=0.7)

        # Final touches on plot
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Day Ahead Spot Price (DKK/MWh)')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # Ensure the directory exists
        os.makedirs(f"{self.path_to_images}/{self.model}/", exist_ok=True)
        plt.savefig(f"{self.path_to_images}/{self.model}/{self.model}_predictions_only_{self.time_period}{self.extra_info}.png", dpi=150, transparent=False)
        plt.show()


    def plot_learning_curves(self, loss, val_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(loss)) + 1, loss, '.', label="Training loss", color='navy')
        plt.plot(np.arange(len(val_loss)) + 1, val_loss, '-', label="Validation loss", color='red')
        plt.legend(fontsize=14)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)

        # Ensure the directory exists
        os.makedirs(f"{self.path_to_images}/{self.model}/", exist_ok=True)
        plt.savefig(f"{self.path_to_images}/{self.model}/{self.model}_learning_curves_{self.time_period}{self.extra_info}.png")
        plt.show()

class LoadData:
    """
    Utility class for loading and preprocessing data.
    """

    def __init__(self, INCLUDE_LAGS=True, INCLUDE_SEASON_VARS=True, INCLUDE_WEATHER=True, TUNER = "Hyperband"):
        self.logger = logging.getLogger()
        self.include_lags = INCLUDE_LAGS
        self.include_season_vars = INCLUDE_SEASON_VARS
        self.include_weather = INCLUDE_WEATHER
        self.tuner = TUNER

    def load_and_preprocess_data(self, data_path, time_start, time_end):
        """
        Load and preprocess the data from a CSV file.

        :param data_path: Path to the CSV file.
        :param time_start: Start time for filtering the dataset.
        :param time_end: End time for filtering the dataset.
        :return: Tuple of preprocessed DataFrame and a formatted time period string.
        """
        try:
            df = pd.read_csv(data_path).dropna().set_index('time')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            if self.include_lags == False:
                lag_columns = [col for col in df.columns if col.startswith('Day Ahead Spot Price_lagged')]
                df = df.drop(lag_columns, axis=1)

            if self.include_season_vars == False:
                weekday_columns = [col for col in df.columns if col.startswith('weekday_')]
                df = df.drop(weekday_columns, axis=1)

            if self.include_weather == False: 
                # Behold kun kolonner, der starter med "Day Ahead Spot Price", "Day Ahead Spot Price_lagged", eller "weekday_"
                columns_to_keep = [col for col in df.columns if col.startswith('Day Ahead Spot Price') or col.startswith('weekday_')]
                df = df[columns_to_keep]

            bool_columns = df.select_dtypes(include='bool').columns
            df[bool_columns] = df[bool_columns].astype(int)
            
            df.rename(columns=lambda x: x.replace("weekday_", ""), inplace=True)
            df.rename(columns=lambda x: x.replace("_", " "), inplace=True)

            time_period = f"{time_start}_to_{time_end}".replace(":", "-")  # Format time period for directory names
            return df.loc[time_start:time_end], time_period
        
        except Exception as e:
            self.logger.error("Failed to load or preprocess data: %s", e)
            raise e

    def setup_directories(self, base_log_dir, base_tuning_dir,  time_period, model):
        """
        Set up directories for logging and tuning, optionally including lags and season variables in directory names.

        :param base_log_dir: Base directory for logs.
        :param base_tuning_dir: Base directory for tuning.
        :param time_period: Time period string for naming.
        :param model: Model name.
        :param include_lags: Whether to include lags in the directory name.
        :param include_season_vars: Whether to include season variables in the directory name.
        :return: Paths for log and tuning directories.
        """
        # Adjust the directory names based on the inclusion of lags and season variables
        extra_info = f'_{self.tuner}'
        if not self.include_lags:
            extra_info += "_ex_lags"
        if not self.include_season_vars:
            extra_info += "_ex_season_dummies"
        if not self.include_weather:
            extra_info += "_ex_wheater"

        # Create the directory paths
        log_dir = os.path.join(base_log_dir, f"{model}/logs_{time_period}{extra_info}")
        tuning_dir = os.path.join(base_tuning_dir, f"{model}/tuning_{time_period}{extra_info}")

        # Make directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tuning_dir, exist_ok=True)

        return log_dir, tuning_dir

        
    def prepare_sequences(self, data, target_column='Day Ahead Spot Price', input_steps=48, output_steps=24, test_size=0.2, val_size=0.5, random_state=42):
        """
        Prepares sequences for training, validation, and testing.

        Parameters:
        - data: DataFrame containing the dataset.
        - target_column: The column name of the target variable.
        - input_steps: Number of input steps (default: 48).
        - output_steps: Number of output steps (default: 24).
        - test_size: Fraction of data to allocate for test + validation (default: 0.2).
        - val_size: Fraction of the test + validation data to allocate for validation (default: 0.5).
        - random_state: Random state for reproducibility (default: 42).

        Returns:
        - sequences_dict: A dictionary containing the scaled sequences for train, validation, and test sets.
        - scalers: A dictionary containing the fitted scalers for features and target variables.
        """
        df = data.copy()
        # Extract features (X) and target variable (y)
        X = df.drop(columns=target_column, axis=1)  # Convert to NumPy array
        y = df[[target_column]]  # Convert to NumPy array

        # Split into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state, shuffle=False)

        # Standardize features and target variable
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)

        X_train_scaled = scaler_X.transform(X_train)
        y_train_scaled = scaler_y.transform(y_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)

        def create_sequences_with_columns(input_x, input_y, input_steps, output_steps, column_names):
            X_seq, y_seq = [], []
            for i in range(len(input_x) - input_steps - output_steps + 1):
                X_seq.append(input_x[i:i + input_steps])  # Collect input sequence
                y_seq.append(input_y[i + input_steps:i + input_steps + output_steps])  # Collect corresponding target sequence
            
            # Convert to 3D arrays while preserving column names
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Re-apply column names to DataFrame for interpretability
            if len(X_seq) > 0:  # Ensure sequence is not empty
                X_seq = [pd.DataFrame(seq, columns=column_names) for seq in X_seq]
            return X_seq, y_seq

        # Example usage
        column_names = X.columns  # Save column names from the original DataFrame

        # Create sequences for training, validation, and testing
        X_train_seq, y_train_seq = create_sequences_with_columns(
            X_train_scaled, y_train_scaled, input_steps, output_steps, column_names)
        X_val_seq, y_val_seq = create_sequences_with_columns(
            X_val_scaled, y_val_scaled, input_steps, output_steps, column_names)
        X_test_seq, y_test_seq = create_sequences_with_columns(
            X_test_scaled, y_test_scaled, input_steps, output_steps, column_names)

        datasets = {
            'X_train': scaler_X.transform(X_train),
            'X_val': scaler_X.transform(X_val),
            'X_test': scaler_X.transform(X_test),
            'y_train': scaler_y.transform(y_train),
            'y_val': scaler_y.transform(y_val),
            'y_test': scaler_y.transform(y_test)
        }
        
        # Store sequences in a dictionary
        sequences_dict = {
            "train": {"X": X_train_seq, "y": y_train_seq},
            "val": {"X": X_val_seq, "y": y_val_seq},
            "test": {"X": X_test_seq, "y": y_test_seq}
        }

        # Store scalers for later inverse transformations
        scalers = {
            "scaler_X": scaler_X,
            "scaler_y": scaler_y
        }

        return sequences_dict, datasets, scalers, X_test_scaled, X

    def split_and_scale_data(self, df):
        """
        Split the dataset into training, validation, and test sets, and scale the features.

        :param df: DataFrame containing the full dataset with 'Day Ahead Spot Price'.
        :return: Tuple of the scaled datasets, target variable, training, and test sets, and scaler for target variable.
        """
        SEED = 1
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        
        X = df.drop('Day Ahead Spot Price', axis=1)
        y = df[['Day Ahead Spot Price']]
            
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

        # Scale data
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)
        datasets = {
            'X_train': scaler_X.transform(X_train),
            'X_val': scaler_X.transform(X_val),
            'X_test': scaler_X.transform(X_test),
            'y_train': scaler_y.transform(y_train),
            'y_val': scaler_y.transform(y_val),
            'y_test': scaler_y.transform(y_test)
        }

        return datasets, y, X_train, X_test, scaler_y, y_test