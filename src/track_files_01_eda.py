import os
import pandas as pd
import time
import os
import pandas as pd
import plotly.express as px
from utils.utils import  *


# Directory to track
directory_to_track = './raw_data'

# Track the files and save the DataFrame
df = track_files_in_directory(directory_to_track)

# Print the DataFrame


# Save the DataFrame to a CSV file
df.to_csv('./file_tracking/tracked_files.csv', index=False)

# Directory to track
directory_to_track = './raw_data'

# Get the file sizes
df = get_file_sizes(directory_to_track)


