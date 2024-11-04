import os
import pandas as pd
import pygwalker as pyg
import plotly.express as px
from pathlib import Path
from datetime import datetime
import streamlit as st
from utils.utils import  create_word_doc,create_rag_report
from openai import OpenAI
import streamlit.components.v1 as components
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from src.build_vector_database_04 import  get_agent
from dotenv import load_dotenv
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from collections import defaultdict
with open('.streamlit/st_auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
st.set_page_config(page_title="File Insights with Pygwalker and Plotly", layout="wide")
# Pre-hashing all plain text passwords once
# Hasher.hash_passwords(config['credentials'])






authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)
if 'information_store' not in st.session_state:
    st.session_state['information_store'] = defaultdict(list)

# Function to accumulate new entries
def store_new_results(prompt, answer, document,all_docs):
    st.session_state['information_store']['input'].append(prompt)
    st.session_state['information_store']['answer'].append(answer)
    st.session_state['information_store']['document'].append(document)
    st.session_state['information_store']['all_docs'].append(all_docs)
authenticator.login()
if st.session_state['authentication_status']:
    authenticator.logout()



    load_dotenv()
    # Set up the Streamlit page

    st.title("File Insights Dashboard")



    agent_executor = get_agent()
    st.session_state['messages']=[]


    st.title("File Share assistant ")
    st.subheader("File Share chat")
    st.divider()
    st.markdown("""
    1. Understanding the RAG Approach:
    RAG is an advanced technique used in modern chatbots where the model first retrieves relevant pieces of information from external sources (documents, databases, or any knowledge repository) and then uses a language model (like GPT) to generate responses based on the retrieved content. This ensures that the chatbot’s responses are both accurate and grounded in factual data.
    
    2. Key Features of the RAG Chatbot:
    Contextual Understanding: The chatbot comprehends your questions in real time and provides responses that are relevant and accurate.
    Document-Driven Responses: It references external documents, ensuring that the answers are grounded in the latest available data.
    Natural Conversation Flow: The chatbot can handle follow-up questions and keeps track of the conversation context.
    Rich Media Support: The chatbot can not only generate textual responses but also provide links to documents, images, or data.
    
    """)
    messages = st.container(height=300)
    if prompt := st.chat_input("Ask the AI Agent anything about the FileShare"):
        messages.chat_message("user").write(prompt)
        with messages.chat_message("assistant"):
            # Initialize the Streamlit callback handler
            st_callback = StreamlitCallbackHandler(st.container())

            # Invoke the agent executor with the callback handler
            response = agent_executor.stream(
                {"input": prompt}, {"callbacks": [st_callback]}
            )

            # Display the response
            for chunk in response:
                print(chunk)
                text = chunk.get("answer", False)
                document= chunk.get("document_names", False)
                if text:
                    st.write(str(text))
                st.session_state.messages.append({"role": "assistant", "content":  text})
                st.divider()
                if document:
                    st.write('Referenzed Chunk of Text:'+str(document))
                    st.divider()
                    st.write('All used Documents retrieved:' + str(set(st.session_state['Documents_used'])))
                    store_new_results(prompt, text, document,list(set(st.session_state['Documents_used']))[:2])

                st.session_state.messages.append({"role": "assistant", "content": document})

    docx_file = create_rag_report(
        st.session_state['information_store']['input'],
        st.session_state['information_store']['answer'],
        st.session_state['information_store']['document'],
        st.session_state['information_store']['all_docs'],
    logo_path = './static/dmu.img',
    intro_image_path = './static/dmu.img'
    )
    # Download button for the user
    st.download_button(
        label="Download Word Document",
        data=docx_file,
        file_name="Q_A_Report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

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


elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
