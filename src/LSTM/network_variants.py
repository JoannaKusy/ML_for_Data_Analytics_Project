import torch
import torch.nn as nn
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Dropout
from keras.regularizers import l1_l2



class LSTMModel0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class keras_LSTM_encoder_decoder:
    """
    Builds a Keras encoder-decoder LSTM functional model.
    Usage:
        builder = keras_LSTM_encoder_decoder(k=lags, n_past_features=..., n_future_features=...)
        model = builder.get_model()
    """
    def __init__(self, input_size, n_past_features, n_future_features,
                 encoder_units=128, decoder_units=128,
                 dense_units=64, dropout=0.2, kernel_regularizer={"l1": 0.01, "l2": 0.01}):
        
        # ----- Encoder -----
        encoder_inputs = Input(shape=(input_size, n_past_features))
        encoder_lstm = LSTM(encoder_units, return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_inputs)  # we only need the states, not the output
        encoder_states = [state_h, state_c]  # pass the states to the decoder as initial state

        # ----- Decoder -----
        decoder_inputs = Input(shape=(1, n_future_features))  # we will feed the decoder one step at a time, so sequence length is 1
        decoder_lstm = LSTM(decoder_units, kernel_regularizer=l1_l2(**kernel_regularizer))
        decoder_output = decoder_lstm(decoder_inputs, initial_state=encoder_states) 

        # ----- Dense head -----
        x = Dense(dense_units, activation='relu')(decoder_output) 
        x = Dropout(dropout)(x)
        output = Dense(1)(x)

        # ----- Model -----
        self.model = Model([encoder_inputs, decoder_inputs], output)

        self.trainable_weights = self.model.trainable_weights

    def get_model(self):
        return self.model

    def __call__(self, inputs):
        return self.model(inputs)
    
    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
    
    def count_params(self):
        return self.model.count_params()
    

