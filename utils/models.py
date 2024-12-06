import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD
import kerastuner as kt
from tensorflow.keras import layers
import numpy as np
from tcn import TCN
from tensorflow import keras
import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from kerastuner.tuners import Hyperband, BayesianOptimization
from utils.misc import TerminateNaN


def create_casual_mask(seq_length):
    """
    Create a causal masking layer for the transformer model.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
    return mask

# Positional Encoding Layer
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model

    def call(self, x):
        seq_length = tf.shape(x)[1]
        angle_rads = self.get_angles(tf.cast(tf.range(seq_length), tf.float32)[:, tf.newaxis], 
                                     tf.cast(tf.range(self.d_model), tf.float32)[tf.newaxis, :], 
                                     tf.cast(self.d_model, tf.float32))

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return x + pos_encoding

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / d_model)
        return pos * angle_rates

class ModelTrainer():
    
    def __init__(self, datasets, log_dir, tuning_dir, loss, model_type='lstm', sequences_dict=None):
        self.model_type = model_type
        self.datasets = datasets
        self.log_dir = log_dir
        self.tuning_dir = tuning_dir
        self.loss = loss
        self.sequences_dict = sequences_dict
        self.output_dim = 24
        self.input_dim = 48

        if model_type.lower() == 'lstm':
            self.build_model = self.build_lstm_model
        elif model_type.lower() == 'tcn':
            self.build_model = self.build_tcn_model
        elif model_type.lower() == 'hybrid':
            self.build_model = self.build_hybrid_model
        elif model_type.lower() == 'transformer':
            self.build_model = self.build_transformer_model


    def build_transformer_model(self, hp):
    # Transformer Encoder with Causal Masking and Positional Encoding
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, mask=None):
            """
            Implements the transformer encoder block with multi-head attention and feed-forward layers.
            A causal mask is applied to ensure the model does not attend to future time steps.
            """
            # Normalization and Multi-Head Self-Attention with Causal Mask
            x = layers.LayerNormalization(epsilon=1e-6)(inputs)
            attention_layer = layers.MultiHeadAttention(
                key_dim=head_size, 
                num_heads=num_heads, 
                dropout=dropout,
                kernel_initializer='glorot_uniform'  # Set initializer for attention layer
            )
            x = attention_layer(x, x, attention_mask=mask)
            x = layers.Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward Network
            x = layers.LayerNormalization(epsilon=1e-6)(res)
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu", 
                            kernel_initializer='he_normal')(x)  # Set initializer for Conv1D layer
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, 
                            kernel_initializer='glorot_uniform')(x)  # Set initializer for second Conv1D
            return x + res

        # Transformer Model Builder with Positional Encoding and Batch-Friendly Mask
        def model_builder(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, pooling_type="average"):
            """
            Builds the Transformer model for time-series forecasting with positional encoding.
            """
            inputs = keras.Input(shape=input_shape)
            x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)  # Dynamic batch size
            seq_length = input_shape[0]  # Assuming (time_steps, features) format
            mask = create_casual_mask(seq_length)  # Create causal mask based on input sequence length and batch size

            # Transformer Encoder Blocks
            for _ in range(num_transformer_blocks):
                x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, mask)

            # Apply pooling based on pooling_type parameter
            if pooling_type == "average":
                x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
            elif pooling_type == "max":
                x = layers.GlobalMaxPooling1D(data_format="channels_last")(x)
            elif pooling_type == "average_max":
                avg_pool = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
                max_pool = layers.GlobalMaxPooling1D(data_format="channels_last")(x)
                x = layers.Concatenate()([avg_pool, max_pool])

            # Fully Connected Layers (MLP)
            for dim in mlp_units:
                x = layers.Dense(dim, activation="relu", kernel_initializer='he_normal')(x)  # Set initializer for Dense layers
                x = layers.Dropout(mlp_dropout)(x)

            # Output layer: Forecasting a single future value (regression)
            outputs = layers.Dense(self.output_dim, kernel_initializer='glorot_uniform')(x)  # Set initializer for output layer
            return keras.Model(inputs, outputs)    

        #datasets = self.sequences_dict
        #self.sequences_dict["train"]["X"] = np.reshape(self.sequences_dict["train"]["X"], (self.sequences_dict["train"]["X"].shape[0], self.sequences_dict["train"]["X"].shape[1], 1))
        #input_shape = (self.sequences_dict["train"]["X"].shape[1], self.sequences_dict["train"]["X"].shape[2])
        input_shape=(self.input_dim, self.sequences_dict["train"]["X"].shape[2])
        head_size = hp.Int('head_size', min_value=64, max_value=512, step=64)
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        ff_dim = hp.Int('ff_dim', min_value=32, max_value=128, step=32)
        num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=4, step=1)
        mlp_units = hp.Int('mlp_units', min_value=64, max_value=512, step=64)
        dropout = hp.Float('dropout', min_value=0, max_value=0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling="log")
        pooling_type = hp.Choice('pooling_type', values=['average', 'max', 'average_max'])

        model = model_builder(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, [mlp_units], dropout, dropout, pooling_type)
        
        optimizer = Adam(learning_rate=learning_rate) if hp.Choice('optimizer', values=['adam', 'sgd']) == 'adam' else SGD(learning_rate=learning_rate)
       
        model.compile(optimizer=optimizer, 
                      loss=self.loss,
                      metrics=['mae'])
        return model


    
    def build_tcn_model(self, hp):
        nb_filters = hp.Int('nb_filters', min_value=32, max_value=128, step=32)
        kernel_size = hp.Int('kernel_size', min_value=2, max_value=8, step=1)
        nb_stacks = hp.Int('nb_stacks', min_value=1, max_value=4, step=1)
        dilations = [1, 2, 4, 8, 16, 32]
        lr = hp.Float('lr_adam', min_value=1e-4, max_value=1e-2, sampling='log')
        dropout_rate = hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.05)
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd'])

        # Configure the optimizer with the learning rate
        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif optimizer_choice == 'sgd':
            optimizer = SGD(learning_rate=lr)
        
        #X_train = np.reshape(self.sequences_dict["train"]["X"], (self.sequences_dict["train"]["X"].shape[0],self.sequences_dict["train"]["X"].shape[1], 1))
        model = Sequential()
        model.add(TCN(
            nb_filters=nb_filters, 
            kernel_size=kernel_size, 
            nb_stacks=nb_stacks, 
            dilations=dilations, 
            padding='causal', 
            use_layer_norm=True,
            use_skip_connections=True, 
            dropout_rate=dropout_rate, 
            return_sequences=False, 
            kernel_initializer='he_normal',  # Apply he_normal initializer
            #input_shape=(X_train.shape[1], 1)
            input_shape=(self.input_dim, self.sequences_dict["train"]["X"].shape[2])
        ))
        model.add(Dense(units=self.output_dim, kernel_initializer='glorot_uniform'))  # Apply glorot_uniform initializer to Dense layer
        
        model.compile(optimizer=optimizer, 
                      loss=self.loss,
                      metrics=['mae'])
        return model

    def build_hybrid_model(self, hp):
        nb_filters = hp.Int('nb_filters', min_value=32, max_value=128, step=32)
        kernel_size = hp.Int('kernel_size', min_value=2, max_value=8, step=1)
        nb_stacks = hp.Int('nb_stacks', min_value=1, max_value=4, step=1)
        dilations = [1, 2, 4, 8, 16, 32]
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
        dense_units = hp.Int('dense_units', min_value=16, max_value=128, step=16)
        dropout_rate = hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.05)
        lr = hp.Float('lr_adam', min_value=1e-4, max_value=1e-2, sampling='log')
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd'])

        # Configure the optimizer with the learning rate
        if optimizer_choice == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif optimizer_choice == 'sgd':
            optimizer = SGD(learning_rate=lr)

        #X_train = np.reshape(self.datasets['X_train'], (self.datasets['X_train'].shape[0], self.datasets['X_train'].shape[1], 1))
        model = Sequential()
        model.add(TCN(
            nb_filters=nb_filters, 
            kernel_size=kernel_size, 
            nb_stacks=nb_stacks, 
            dilations=dilations,
            padding='causal', 
            use_layer_norm=True, 
            use_skip_connections=True, 
            dropout_rate=dropout_rate, 
            return_sequences=True, 
            kernel_initializer='he_normal',  # Set initializer for TCN layer
            #input_shape=(X_train.shape[1], 1)
            input_shape=(self.input_dim, self.sequences_dict["train"]["X"].shape[2])
        ))
        model.add(LSTM(
            units=lstm_units, 
            return_sequences=False, 
            kernel_initializer='orthogonal'  # Set initializer for LSTM layer
        ))
        model.add(Dropout(dropout_rate))
        model.add(Dense(
            units=dense_units, 
            activation='relu', 
            kernel_initializer='he_normal'  # Set initializer for Dense layer with ReLU activation
        ))
        model.add(Dense(
            units=self.output_dim, 
            kernel_initializer='glorot_uniform'  # Set initializer for output layer
        ))

        model.compile(optimizer=optimizer, 
                      loss=self.loss,
                      metrics=['mae'])
        return model
    
  

    def build_lstm_model(self, hp):
        model = Sequential([
            LSTM(
                units=hp.Int('units', min_value=32, max_value=128, step=32),
                return_sequences=True,
                input_shape=(self.input_dim, self.sequences_dict["train"]["X"].shape[2]),
                kernel_initializer='orthogonal'  # Set the initializer here
            ),
            Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)),
            LSTM(
                units=hp.Int('units', min_value=32, max_value=128, step=32),
                return_sequences=False,
                kernel_initializer='orthogonal'  # Apply the initializer to other layers if needed
            ),
            Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)),
            Dense(
                units=hp.Int('dense_units', min_value=16, max_value=128, step=16),
                activation='relu',
                kernel_initializer='he_normal'  # Apply initializer to Dense layers as well
            ),
            Dense(self.output_dim)  # Output layer with 24 units
        ])
        
        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
            loss='mean_absolute_error',
            metrics=['mae']
        )
        return model

    def train_model(self, batch_size=32, training_epoch=30, final_model_epoch=50, overwrite=False, bayesian=False):

            start_time = time.time()

            #min_epochs_callback = MinimumEpochTrialCallback(min_epochs=15)

            for split in ["train", "val", "test"]:
                self.sequences_dict[split]["X"] = np.array(self.sequences_dict[split]["X"])
                self.sequences_dict[split]["y"] = np.array(self.sequences_dict[split]["y"])

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-5)
            terminate_nan = TerminateNaN()

            tuner = Hyperband(
                lambda hp: self.build_model(hp),
                objective='val_loss',
                max_epochs=training_epoch,
                directory=self.tuning_dir,
                factor=7,
                overwrite=overwrite,
                project_name=f'my_{self.model_type}_model_project',
                )

            if bayesian:
                tuner = kt.BayesianOptimization(
                    self.build_model,
                    objective='val_loss',
                    max_trials=training_epoch,
                    num_initial_points=5,
                    directory=self.tuning_dir,
                    project_name=f'my_{self.model_type}_model_project')  
            
            evaluation_interval = int(np.ceil(np.array(self.sequences_dict["train"]["X"]).shape[0] / batch_size))

            tuner.search(self.sequences_dict["train"]["X"], self.sequences_dict["train"]["y"],
                        epochs=training_epoch,
                        steps_per_epoch=evaluation_interval,
                        validation_data=(self.sequences_dict["val"]["X"], self.sequences_dict["val"]["y"]),
                        callbacks=[early_stopping, reduce_lr, terminate_nan])

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            final_model = tuner.hypermodel.build(best_hps)
            final_model.summary()

            history = final_model.fit(self.sequences_dict["train"]["X"], self.sequences_dict["train"]["y"],
                                    batch_size=batch_size, 
                                    epochs=final_model_epoch,
                                    validation_data=(self.sequences_dict["val"]["X"], self.sequences_dict["val"]["y"]),
                                    callbacks=[reduce_lr])
            
            # Calculate the total time taken
            end_time = time.time()
            duration = end_time - start_time

            return final_model, history, best_hps, duration
    

