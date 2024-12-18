import pandas as pd
import matplotlib.pyplot as plt
import os

# Set up the directory path
directory = 'Graph'

# Initialize an empty list to hold the dataframes
dfs = []

# Loop through the subdirectories in the directory
for subdir in os.listdir(directory):
    if os.path.isdir(os.path.join(directory, subdir)):
        # Loop through the files in the subdirectory
        for filename in os.listdir(os.path.join(directory, subdir)):
            if filename.endswith('.csv'):
                # Read in the csv file as a dataframe
                df = pd.read_csv(os.path.join(directory, subdir, filename))
                # Add the dataframe to the list
                dfs.append(df)

# Concatenate the dataframes along the column axis
df_all = pd.concat(dfs, keys=range(len(dfs)), names=['File']).reset_index()

# Set up the line plot
plt.figure(figsize=(10, 5))
for i, d in df_all.groupby('File'):
    plt.plot(d['Time'], d['Value'], label=f'File {i}')

# Set the x and y labels
plt.xlabel('Time')
plt.ylabel('Value')

# Add a legend
plt.legend()

# Show the plot
plt.show()
