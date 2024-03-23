import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

# Function to load and process data
@st.cache
def load_data():
    # Adjust the path if your file is located elsewhere
    data = pd.read_csv('/path/to/Connections.csv', parse_dates=['Connected On'])
    data['YearMonth'] = data['Connected On'].dt.to_period('M')
    return data

# Function to calculate connections
def calculate_connections(data, period):
    end_date = data['Connected On'].max()
    if period == 'month':
        start_date = end_date - pd.DateOffset(months=1)
    else:  # year
        start_date = end_date - pd.DateOffset(years=1)
    return data[(data['Connected On'] >= start_date) & (data['Connected On'] <= end_date)].shape[0]

# Function to train model and make predictions
def predict_growth(data, months_ahead):
    # Preparing data for the model
    data = data.groupby('YearMonth').size().reset_index(name='Connections')
    data['Time'] = np.arange(len(data))
    
    # Train the model
    model = LinearRegression()
    model.fit(data[['Time']], data['Connections'])

    # Making predictions
    max_time = data['Time'].max()
    future_times = np.array([max_time + i for i in range(1, months_ahead + 1)]).reshape(-1, 1)
    predictions = model.predict(future_times)
    return predictions

# Streamlit UI
def main():
    st.title('LinkedIn Connections Analysis')

    data = load_data()

    # Display connections made in the last month and year
    st.write(f"Connections in the last month: {calculate_connections(data, 'month')}")
    st.write(f"Connections in the last year: {calculate_connections(data, 'year')}")

    # Predict future connection growth
    if st.button('Predict Future Connections'):
        months_ahead = st.selectbox('Select Period', [1, 12], format_func=lambda x: 'Next Month' if x == 1 else 'Next Year')
        predictions = predict_growth(data, months_ahead)
        st.write(f"Predicted connections in the next {'month' if months_ahead == 1 else 'year'}: {int(predictions[-1])}")

        # Plotting
        plt.figure()
        plt.plot(data['YearMonth'], data.groupby('YearMonth').size(), label='Historical Data')
        future_months = pd.date_range(data['Connected On'].max(), periods=months_ahead + 1, freq='M').to_period('M')[1:]
        plt.plot(future_months, predictions, label='Predictions')
        plt.xticks(rotation=45)
        plt.xlabel('Time')
        plt.ylabel('Connections')
        plt.title('LinkedIn Connections Growth Prediction')
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt.gcf())

if __name__ == "_main_":
    main()