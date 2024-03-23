import streamlit as st
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def load_json(file):
    """
    Load JSON file.
    """
    return json.load(file)

def process_data(data):
    """
    Extract and process timestamps and names from the data.
    """
    timestamps = []
    names = []
    for item in data:
        timestamps.append(item['string_list_data'][0]['timestamp'])
        names.append(item['string_list_data'][0]['value'])
    dates = [datetime.fromtimestamp(ts) for ts in timestamps]
    return dates, names

def analyze_growth(dates):
    """
    Analyze growth based on the dates.
    """
    dates.sort()
    followers_count = list(range(1, len(dates) + 1))
    return pd.DataFrame({'Date': dates, 'Followers': followers_count})

def plot_growth(df):
    """
    Plot the growth of followers over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Followers'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Number of Followers')
    plt.title('Followers Growth Over Time')
    plt.grid(True)
    st.pyplot(plt)

def calculate_growth_rate(df):
    """
    Calculate the rate of growth of followers.
    """
    time_diff = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    growth_rate = (df['Followers'].iloc[-1] - df['Followers'].iloc[0]) / time_diff
    return growth_rate

def predict_followers(df):
    """
    Predict future followers using linear regression.
    """
    df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())
    X = df[['Date_Ordinal']]
    y = df['Followers']

    model = LinearRegression()
    model.fit(X, y)

    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date, periods=365)
    future_dates_ordinal = [d.toordinal() for d in future_dates]

    predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1))
    return future_dates, predictions

# Streamlit interface
st.title('Instagram Followers Growth Analysis')

# File uploader for followers JSON file
followers_file = st.file_uploader("Upload Followers JSON File", type="json")
if followers_file is not None:
    # Load and process the data
    followers_data = load_json(followers_file)
    dates, names = process_data(followers_data)
    df = analyze_growth(dates)

    # Displaying follower names
    st.subheader('List of Followers')
    st.write(names)

    # Display the count and date range of followers
    num_followers = len(names)
    st.markdown(f"*Total Number of Followers Received:* {num_followers}")
    st.markdown(f"*Date Range:* {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

    # Display followers growth data in a table
    st.subheader('Followers Growth Data')
    st.write(df)

    # Plotting the followers growth chart
    st.subheader('Followers Growth Chart')
    plot_growth(df)

    # Calculating and displaying the rate of growth
    growth_rate = calculate_growth_rate(df)
    st.markdown(f"*Average Rate of Growth:* {growth_rate:.2f} followers per day")

    # Predicting followers growth for the next year
    st.subheader('Predicted Followers Growth for the Next Year')
    future_dates, predictions = predict_followers(df)
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Followers': predictions})

    # Display the predicted growth chart
    st.line_chart(pred_df.set_index('Date'))

    # Displaying prediction details
    predicted_growth_rate = (predictions[-1] - df['Followers'].iloc[-1]) / 365
    st.markdown(f"*Predicted Growth Rate:* {predicted_growth_rate:.2f} followers per day")
    st.markdown(f"*Predicted Total Followers in 1 Year:* {int(predictions[-1])}")