import numpy as np
import pandas as pd
import streamlit as st

def display():
    st.title("Throughput Analysis")
    st.write("This is the Throughput Analysis page.")
    
    # Load the data
    data = pd.read_csv('./notebooks/Week2_data.csv')

    # Display the first few rows
    st.write("Data Preview")
    st.write(data.head())

    # Perform Throughput calculations
    data['Throughput'] = (data['Total DL (Bytes)'] + data['Total UL (Bytes)']) / data['Dur. (ms)']
    
    # Display top 10 and bottom 10 throughput values
    top_throughput = data['Throughput'].nlargest(10)
    bottom_throughput = data['Throughput'].nsmallest(10)
    
    st.write("Top 10 Throughput Values:")
    st.write(top_throughput)
    
    st.write("Bottom 10 Throughput Values:")
    st.write(bottom_throughput)
    
    # Display most frequent throughput values
    most_frequent_throughput = data['Throughput'].value_counts().nlargest(10)
    st.write("Most Frequent 10 Throughput Values:")
    st.write(most_frequent_throughput)
    
    # Calculate RTT and display
    data['Start'] = pd.to_datetime(data['Start'])
    data['End'] = pd.to_datetime(data['End'])
    data['RTT'] = (data['End'] - data['Start']).dt.total_seconds() * 1000  # Convert seconds to milliseconds
    
    average_rtt = data['RTT'].mean()
    st.write(f'Average RTT: {average_rtt} ms')

    # Check for Packet_Loss column and estimate retransmissions
    if 'Packet_Loss' in data.columns:
        average_packet_size = 1500  # Bytes per packet
        data['Total_Packets_Sent'] = (data['Total DL (Bytes)'] + data['Total UL (Bytes)']) / average_packet_size
        data['Estimated_Retransmissions'] = data['Total_Packets_Sent'] * (data['Packet_Loss'] / 100)
        
        average_retransmission = data['Estimated_Retransmissions'].mean()
        st.write(f'Average Estimated TCP Retransmission Rate: {average_retransmission}')
    else:
        st.write('Packet_Loss column is missing in the dataset.')

    # Plot throughput using Streamlit line chart
    st.line_chart(data['Throughput'])

# Call the display function to render the dashboard
if __name__ == '__main__':
    display()


