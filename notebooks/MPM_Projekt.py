#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd 
import plotly.express as px
from plotly.offline import init_notebook_mode
import os
import time
init_notebook_mode()


# In[93]:


data=pd.read_csv('tracked_files.csv')

from pathlib import Path
from datetime import datetime

def get_file_modified_time(file_path):
    # Get file metadata
    file_info = Path(file_path)
    
    # Get the modification time (in seconds since epoch)
    modified_time = file_info.stat().st_mtime
    
    # Convert to a human-readable format
    modified_time = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
    
    return modified_time


# In[94]:


def get_file_sizes(directory):
    file_data = []

    # Walk through the directory and collect file information
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)  # Get file size in bytes
            relative_path = os.path.relpath(file_path, directory)  # Get relative file path

            file_data.append({
                'File Name': file,
                'File Path': relative_path,
                'File Size (Bytes)': file_size,
                'Directory': os.path.basename(root),  # Directory name
               'Modified Time':get_file_modified_time(file_path)  # Ge
            })

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(file_data)
    return df

def create_treemap(df):
    # Create a Plotly treemap with file sizes
    fig = px.treemap(
        df,
        path=['Directory', 'File Name'],  # Hierarchical view
        values='File Size (Bytes)',  # Size of boxes based on file size
        title='File Sizes Treemap',
        color='File Size (Bytes)',  # Color based on file size
        hover_data={'File Size (Bytes)': ':.2f'}  # Show file size on hover
    )
    fig.update_layout(height=1000)
    return fig


# In[95]:


df=get_file_sizes('./data_store')
create_treemap(df)


# In[96]:


df['folder_name']=df['File Path'].str.split('/',expand=True)[0]
df['subfolder_name']=df['File Path'].str.split('/',expand=True)[1]


# In[97]:


df.sort_values(by=['File Size (Bytes)']).head()


# In[108]:


df['Modified Time']=pd.to_datetime(df['Modified Time'])


# In[109]:


px.bar(df.groupby('Directory')['File Name'].count().reset_index(),
       x='Directory',y='File Name',color='Directory',title='File Counts per Folder',height=800)


# In[110]:


px.bar(df.groupby('Directory')[ 'File Size (Bytes)'].sum().reset_index(),
       x='Directory',y='File Size (Bytes)',color='Directory',title='File Size per Folder',height=800)


# In[111]:


px.line(df.sort_values(by=['File Size (Bytes)'],ascending=False)\
        .reset_index()['File Size (Bytes)'].cumsum(),title='Cummulative File Size Distribution',height=800)


# In[142]:


df['Modifiedweek']=  df['Modified Time'].dt.isocalendar().week
df['Modifiedyear']=  df['Modified Time'].dt.year 


# In[156]:


px.bar(df.groupby(['Modifiedyear','Modifiedweek'])['File Name'].count().reset_index().\
assign(Kalender_Week= lambda df: 'KW_' + df.Modifiedweek.astype(str) + '_' + df.Modifiedyear.astype(str)	 ).rename(columns={'File Name':'File_Count'}),height=800,y='Kalender_Week',x='File_Count')


# In[201]:


df['Modifiedmonth']=  df['Modified Time'].dt.month
def get_first_day_of_week(year, week):
    # Get the Monday of the given calendar week
    return datetime.strptime(f'{year} {week} 1', '%G %V %u').date()


# In[190]:


src=pd.concat([df.groupby(['folder_name','Modifiedyear','Modifiedweek'])['File Size (Bytes)'].sum().reset_index()\
.groupby('folder_name')['File Size (Bytes)'].expanding().sum().reset_index(),df.groupby(['folder_name','Modifiedyear','Modifiedweek'])['File Size (Bytes)'].sum()\
          .reset_index()[['Modifiedyear','Modifiedweek']]],axis=1)


# In[203]:


src['date']=src.apply(lambda d: get_first_day_of_week(d.loc['Modifiedyear'],d.loc['Modifiedweek']),axis=1)


# In[210]:


px.area(src,x='date',y='File Size (Bytes)',color='folder_name',title='Selective Cummultaive Sum Per Folder',height=800)


# In[ ]:




