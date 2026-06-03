import torch
import torch.nn as nn
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Dropout
from keras.regularizers import l1_l2


class LSTMModel0(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        n_future_features=None,
        n_past_features=None,
    ):
        super().__init__()
        # input_size to match n_past_features from create_sequences
        self.lstm = nn.LSTM(
            input_size=n_past_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_past, x_future=None):
        # added x_future=None to maintain compatibility with direct.py
        out, _ = self.lstm(x_past)
        out = out[:, -1, :]  # the last hidden state
        return self.fc(out)


class LSTMAttentionModel(nn.Module):
    def __init__(
        self,
        input_size,
        n_future_features,
        hidden_size,
        num_layers,
        dropout,
        n_past_features=None,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_past_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size + n_future_features, 1)

    def forward(self, x_past, x_future, return_attn=False):
        # x_past shape: (batch, sequence_length, n_past_features)
        lstm_out, _ = self.lstm(x_past)

        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.sum(attn_weights * lstm_out, dim=1)

        combined = torch.cat((context, x_future.view(x_future.size(0), -1)), dim=1)
        out = self.fc(combined)

        if return_attn:
            return out, attn_weights
        return out


class keras_LSTM_encoder_decoder:
    """
    Builds a Keras encoder-decoder LSTM functional model.
    Usage:
        builder = keras_LSTM_encoder_decoder(k=lags, n_past_features=..., n_future_features=...)
        model = builder.get_model()
    """

    def __init__(
        self,
        input_size,
        n_past_features,
        n_future_features,
        encoder_units=128,
        decoder_units=128,
        dense_units=64,
        dropout=0.2,
        kernel_regularizer={"l1": 0.01, "l2": 0.01},
    ):

        # ----- Encoder -----
        encoder_inputs = Input(shape=(input_size, n_past_features))
        encoder_lstm = LSTM(encoder_units, return_state=True)
        _, state_h, state_c = encoder_lstm(
            encoder_inputs
        )  # we only need the states, not the output
        encoder_states = [
            state_h,
            state_c,
        ]  # pass the states to the decoder as initial state

        # ----- Decoder -----
        decoder_inputs = Input(
            shape=(1, n_future_features)
        )  # we will feed the decoder one step at a time, so sequence length is 1
        decoder_lstm = LSTM(
            decoder_units, kernel_regularizer=l1_l2(**kernel_regularizer)
        )
        decoder_output = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        # ----- Dense head -----
        x = Dense(dense_units, activation="relu")(decoder_output)
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
