import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Title of the app
st.title('Stock Price Prediction')

# Sidebar inputs
option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()
duration = st.sidebar.number_input('Enter duration (days)', value=3000, min_value=1)
today = datetime.date.today()
start_date = st.sidebar.date_input('Start Date', value=today - datetime.timedelta(days=duration))
end_date = st.sidebar.date_input('End Date', value=today)

# Download stock data
@st.cache_resource
def download_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date, progress=False)

if start_date < end_date:
    data = download_data(option, start_date, end_date)
else:
    st.sidebar.error('Error: End date must be after start date.')

scaler = StandardScaler()

# Main function to manage app flow
def main():
    option = st.sidebar.selectbox('Choose an option', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        visualize_technical_indicators()
    elif option == 'Recent Data':
        show_recent_data()
    elif option == 'Predict':
        prediction_interface()

# Function to visualize technical indicators
def visualize_technical_indicators():
    st.header('Technical Indicators')
    indicator = st.radio('Select a Technical Indicator', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Compute indicators
    bb_indicator = BollingerBands(data['Close'])
    data['BB_High'] = bb_indicator.bollinger_hband()
    data['BB_Low'] = bb_indicator.bollinger_lband()
    data['MACD'] = MACD(data['Close']).macd()
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['SMA'] = SMAIndicator(data['Close'], window=14).sma_indicator()
    data['EMA'] = EMAIndicator(data['Close']).ema_indicator()

    # Visualize based on selection
    if indicator == 'Close':
        st.line_chart(data['Close'])
    elif indicator == 'BB':
        st.line_chart(data[['Close', 'BB_High', 'BB_Low']])
    elif indicator == 'MACD':
        st.line_chart(data['MACD'])
    elif indicator == 'RSI':
        st.line_chart(data['RSI'])
    elif indicator == 'SMA':
        st.line_chart(data['SMA'])
    elif indicator == 'EMA':
        st.line_chart(data['EMA'])

# Function to show recent data
def show_recent_data():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

# LSTM model function
def lstm_model():
    st.header('LSTM Model for Prediction')
    # Example of two different number_inputs with unique keys
    num_days = st.number_input(
        'Number of days to forecast', 
        value=5, 
        min_value=1, 
        key='num_days_forecast'
    )

    num_epochs = st.number_input(
        'Number of epochs for training', 
        value=10, 
        min_value=1, 
        key='num_epochs_training'
    )

    df = data[['Close']]
    df['Target'] = df['Close'].shift(-num_days)

    x = df.drop(['Target'], axis=1).iloc[:-num_days]
    y = df['Target'].dropna()

    x_scaled = scaler.fit_transform(x)
    x_scaled = x_scaled.reshape((x_scaled.shape[0], 1, 1))

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=7)

    # LSTM Model
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)),
        Dropout(0.2),
        LSTM(60, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(80, activation='relu', return_sequences=True),
        Dropout(0.4),
        LSTM(120, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

    preds = model.predict(x_test)
    x_forecast = scaler.transform(df[['Close']].iloc[-num_days:].values).reshape((num_days, 1, 1))
    forecast = model.predict(x_forecast)

    st.write(f'R2 Score: {r2_score(y_test, preds)}')
    st.write(f'MAE: {mean_absolute_error(y_test, preds)}')

    st.header('Forecast for Next Days')
    for i, pred in enumerate(forecast.flatten(), 1):
        st.write(f'Day {i}: {pred:.2f}')

# General prediction interface
def prediction_interface():
    model_choice = st.radio('Select a model', ['Linear Regression', 'Random Forest', 'Extra Trees', 'K-Nearest Neighbors', 'XGBoost', 'LSTM'])
    num_days = st.number_input('Number of days to forecast', value=5, min_value=1)

    if st.button('Predict'):
        if model_choice == 'LSTM':
            lstm_model()
        else:
            sklearn_model_engine(model_choice, num_days)

# Function to train and forecast with sklearn models
def sklearn_model_engine(model_choice, num_days):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Extra Trees': ExtraTreesRegressor(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'XGBoost': XGBRegressor()
    }

    model = models[model_choice]
    df = data[['Close']]
    df['Target'] = df['Close'].shift(-num_days)

    x = scaler.fit_transform(df[['Close']].iloc[:-num_days])
    y = df['Target'].dropna()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    forecast = model.predict(scaler.transform(df[['Close']].iloc[-num_days:].values))

    st.write(f'R2 Score: {r2_score(y_test, preds)}')
    st.write(f'MAE: {mean_absolute_error(y_test, preds)}')

    st.header('Forecast for Next Days')
    for i, pred in enumerate(forecast, 1):
        st.write(f'Day {i}: {pred:.2f}')

if __name__ == '__main__':
    main()
