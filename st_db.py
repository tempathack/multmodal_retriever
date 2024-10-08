import os
import pandas as pd
import pygwalker as pyg
import plotly.express as px
from pathlib import Path
from datetime import datetime
import streamlit as st
from openai import OpenAI
import streamlit.components.v1 as components
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from src.build_vector_database_04 import  get_agent
from dotenv import load_dotenv


load_dotenv()
# Set up the Streamlit page
st.set_page_config(page_title="File Insights with Pygwalker and Plotly", layout="wide")
st.title("File Insights Dashboard")



agent_executor = get_agent()
# Function to get the modified time of a file
def get_file_modified_time(file_path):
    file_info = Path(file_path)
    modified_time = file_info.stat().st_mtime
    return datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')

# Function to gather file sizes and metadata
def get_file_sizes(directory):
    file_data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            relative_path = os.path.relpath(file_path, directory)
            file_data.append({
                'File Name': file,
                'File Path': relative_path,
                'File Size (Bytes)': file_size,
                'Directory': os.path.basename(root),
                'Modified Time': get_file_modified_time(file_path)
            })
    return pd.DataFrame(file_data)

# Directory to scan for files
directory = './raw_data'
df = get_file_sizes(directory)

# Convert modified time to datetime and create additional columns
df['Modified Time'] = pd.to_datetime(df['Modified Time'])
df['folder_name'] = df['File Path'].str.split('/', expand=True)[0]
df['subfolder_name'] = df['File Path'].str.split('/', expand=True)[1]
df['Modifiedweek'] = df['Modified Time'].dt.isocalendar().week
df['Modifiedyear'] = df['Modified Time'].dt.year
df['Modifiedmonth'] = df['Modified Time'].dt.month

# Pygwalker Visualization
pyg_html = pyg.to_html(df)
components.html(pyg_html, height=1000, scrolling=True)
st.divider()
# Create Plotly plots
st.subheader("Treemap of File Sizes")
treemap_fig = px.treemap(
    df,
    path=['Directory', 'File Name'],
    values='File Size (Bytes)',
    title='File Sizes Treemap',
    color='File Size (Bytes)',
    hover_data={'File Size (Bytes)': ':.2f'},height=800,
)
st.plotly_chart(treemap_fig)

# Bar chart: File counts per folder
st.subheader("File Counts per Folder")
file_count_fig = px.bar(
    df.groupby('Directory')['File Name'].count().reset_index(),
    x='Directory', y='File Name', color='Directory', title='File Counts per Folder', height=800
)
st.plotly_chart(file_count_fig)

# Bar chart: File size per folder
st.subheader("File Size per Folder")
file_size_fig = px.bar(
    df.groupby('Directory')['File Size (Bytes)'].sum().reset_index(),
    x='Directory', y='File Size (Bytes)', color='Directory', title='File Size per Folder', height=800
)
st.plotly_chart(file_size_fig)

# Line chart: Cumulative file size distribution
st.subheader("Cumulative File Size Distribution")
cumulative_size_fig = px.line(
    df.sort_values(by=['File Size (Bytes)'], ascending=False).reset_index()['File Size (Bytes)'].cumsum(),
    title='Cumulative File Size Distribution', height=800
)
st.plotly_chart(cumulative_size_fig)

# Area chart: Selective cumulative sum per folder
st.subheader("Cumulative File Size Per Folder Over Time")
src = pd.concat([
    df.groupby(['folder_name', 'Modifiedyear', 'Modifiedweek'])['File Size (Bytes)'].sum().reset_index().groupby('folder_name')['File Size (Bytes)'].expanding().sum().reset_index(),
    df.groupby(['folder_name', 'Modifiedyear', 'Modifiedweek'])['File Size (Bytes)'].sum().reset_index()[['Modifiedyear', 'Modifiedweek']]
], axis=1)

# Function to calculate the first day of the week
def get_first_day_of_week(year, week):
    return datetime.strptime(f'{year} {week} 1', '%G %V %u').date()

src['date'] = src.apply(lambda d: get_first_day_of_week(d['Modifiedyear'], d['Modifiedweek']), axis=1)

cumulative_folder_fig = px.area(
    src, x='date', y='File Size (Bytes)', color='folder_name', title='Cumulative Sum Per Folder', height=800
)
st.plotly_chart(cumulative_folder_fig)

# Bar chart: File count by calendar week
st.subheader("File Count by Calendar Week")
file_count_week_fig = px.bar(
    df.groupby(['Modifiedyear', 'Modifiedweek'])['File Name'].count().reset_index().rename(columns={'File Name': 'File_Count'}),
    x='Modifiedweek',  # Use calendar week on the x-axis
    y='File_Count',  # File count on the y-axis
    height=800,
    facet_col='Modifiedyear',  # Create facet columns for each year
    title='File Counts per Calendar Week, Faceted by Year'
)

# Update x-axis title
file_count_week_fig.update_xaxes(title_text='Calendar Week')

# Show the chart in Streamlit
st.plotly_chart(file_count_week_fig)

st.divider()

st.title("File Share assistant ")

import streamlit as st

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to handle chat actions
def chat_actions():
    user_message = st.session_state["chat_input"]

    # Append user message to chat history
    st.session_state["chat_history"].append(
        {"role": "user", "content": user_message},
    )

    # Initialize the Streamlit callback handler
    st_callback = StreamlitCallbackHandler(st.container())

    # Invoke the agent executor with the user input
    response = agent_executor.stream(
        {"input": user_message}, {"callbacks": [st_callback]}
    )

    # Process the response
    for chunk in response:
        text = chunk.get("answer", "")
        document = chunk.get("document_names", "")

        # Optionally append additional information (documents, references)
        if document!='' or text!='':
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": text},
            )

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": f"Referenz Document: {document}"}
            )
        st.divider()

    # Clear the input field
    st.session_state["chat_input"] = ""

# Chat input field
st.text_input("Enter your message", key="chat_input", on_change=chat_actions)

# Display chat history
for message in st.session_state["chat_history"]:
    with st.chat_message(name=message["role"]):
        st.write(message["content"])
