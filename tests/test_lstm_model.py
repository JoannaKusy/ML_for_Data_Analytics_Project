import torch
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from LSTM.network_variants import LSTMAttentionModel

def test_lstm_forward_pass_shape():
    """Test that the model accepts batched input and returns the correct output shape."""
    # dummy configurations based on dataset
    batch_size = 16
    n_past_features = 12
    n_future_features = 4
    hidden_size = 64
    
    # (batch_size, sequence_length, features) - assuming sequence_length of 1 
    x_past = torch.randn(batch_size, 1, n_past_features)
    x_future = torch.randn(batch_size, 1, n_future_features)
    
    model = LSTMAttentionModel(
        input_size=1, 
        n_future_features=n_future_features, 
        hidden_size=hidden_size, 
        num_layers=1, 
        dropout=0.0,
        n_past_features=n_past_features
    )
    
    model.eval()
    with torch.no_grad():
        output = model(x_past, x_future)
        
    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs!"