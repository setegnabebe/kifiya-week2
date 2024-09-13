import pandas as pd
import numpy as np
from scipy.stats import zscore
import streamlit as st

# Function to handle outliers
def handle_outliers(df, column):
    column_data = df[column].dropna()
    
    z_scores = zscore(column_data)
    abs_z_scores = np.abs(z_scores)
    
    # Find the indices where the z-score is greater than 3 (outliers)
    outlier_indices = np.where(abs_z_scores > 3)
    
    # Replace outliers with the mean of the column
    df.loc[df.index[outlier_indices[0]], column] = df[column].mean()

# Streamlit app to display the Clustering Analysis
def display():
    st.title("Clustering Analysis")
    st.write("This is the Clustering Analysis page.")
    
    # Load the data
    df = pd.read_csv('./notebooks/Week2_data.csv')
    
    # Show the raw data before handling missing values
    st.write("Raw Data Preview:")
    st.write(df.head())
    
    # Handle missing values
    df['TCP DL Retrans. Vol (Bytes)'] = df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean())
    df['TCP UL Retrans. Vol (Bytes)'] = df['TCP UL Retrans. Vol (Bytes)'].fillna(df['TCP UL Retrans. Vol (Bytes)'].mean())
    df['Avg RTT DL (ms)'] = df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean())
    df['Avg RTT UL (ms)'] = df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean())
    df['Avg Bearer TP DL (kbps)'] = df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean())
    df['Avg Bearer TP UL (kbps)'] = df['Avg Bearer TP UL (kbps)'].fillna(df['Avg Bearer TP UL (kbps)'].mean())
    df['Handset Type'] = df['Handset Type'].fillna(df['Handset Type'].mode()[0])
    
    # Display the data after handling missing values
    st.write("Data After Handling Missing Values:")
    st.write(df.head())
    
    # Handle outliers in relevant columns
    handle_outliers(df, 'TCP DL Retrans. Vol (Bytes)')
    handle_outliers(df, 'TCP UL Retrans. Vol (Bytes)')
    handle_outliers(df, 'Avg RTT DL (ms)')
    handle_outliers(df, 'Avg RTT UL (ms)')
    handle_outliers(df, 'Avg Bearer TP DL (kbps)')
    handle_outliers(df, 'Avg Bearer TP UL (kbps)')
    
    # Display data after handling outliers
    st.write("Data After Handling Outliers:")
    st.write(df.head())
    
    # Create visualizations
    st.write("TCP DL Retransmission Volume Distribution:")
    st.bar_chart(df['TCP DL Retrans. Vol (Bytes)'])

    st.write("Avg RTT DL (ms) Distribution:")
    st.line_chart(df['Avg RTT DL (ms)'])

# Call the display function to render the dashboard
if __name__ == '__main__':
    display()
