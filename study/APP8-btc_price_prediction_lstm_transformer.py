
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

# 1. 데이터 로딩 및 전처리
df = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01')[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def make_dataset(data, window_size=60):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 60
X, y = make_dataset(scaled_data, window_size)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2. LSTM 모델 정의 및 학습
lstm_model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(window_size, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

# 3. LSTM 예측 결과 시각화
predicted_lstm = lstm_model.predict(X_test)
predicted_lstm = scaler.inverse_transform(predicted_lstm)
actual = scaler.inverse_transform(y_test)

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual')
plt.plot(predicted_lstm, label='LSTM Predicted')
plt.title("LSTM-based BTC Price Prediction")
plt.legend()
plt.show()

# 4. Transformer 모델 정의
def transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

# 5. Transformer 학습
transformer = transformer_model((window_size, 1))
transformer.compile(optimizer=Adam(), loss='mse')
transformer.fit(X_train, y_train, epochs=10, batch_size=32)

# 6. Transformer 예측 시각화
predicted_transformer = transformer.predict(X_test)
predicted_transformer = scaler.inverse_transform(predicted_transformer)

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual')
plt.plot(predicted_transformer, label='Transformer Predicted')
plt.title("Transformer-based BTC Price Prediction")
plt.legend()
plt.show()
