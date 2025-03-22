import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of CSV file paths
file_paths = [
    "April_saccade_summary.csv",
    "February_saccade_summary.csv",
    "January_saccade_summary.csv",
    "June_saccade_summary.csv",
    "March_saccade_summary.csv",
    "May_saccade_summary.csv",
    "2022_saccade_summary.csv",
    "2023_7_12_saccade_summary.csv"
]

# 1. Concatenate all CSV files into a giant dataframe.
df_list = []
for file in file_paths:
    if os.path.exists(file):
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    else:
        print(f"File not found: {file}")

# Concatenate dataframes (reset index so that it’s continuous)
concatenated_df = pd.concat(df_list, ignore_index=True)

# Save the concatenated dataframe to a CSV file
output_csv = "saccade_summary.csv"
concatenated_df.to_csv(output_csv, index=False)
print(f"Concatenated dataframe saved to {output_csv}")

# Set seaborn style for all plots
sns.set(style="whitegrid")

# 2. Plotting distributions

# Function to save and show each plot
def save_and_show(plot_filename):
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

# Plot distribution for "age" (assumed numeric)
plt.figure(figsize=(8, 6))
sns.histplot(concatenated_df['age'], kde=True)
plt.xlabel("Age", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.title("Distribution of Age", fontsize=20)
plt.legend(labels=["age"])
save_and_show("distribution_age.png")

# Plot distribution for "WorkstationName" (categorical: count plot)
plt.figure(figsize=(8, 6))
sns.countplot(data=concatenated_df, x='WorkstationName')
plt.xlabel("WorkstationName", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Distribution of WorkstationName", fontsize=20)
plt.legend(labels=["WorkstationName"])
plt.xticks(rotation=45)
save_and_show("distribution_WorkstationName.png")

# Plot distribution for "WorkstationName_video" (categorical: count plot)
plt.figure(figsize=(8, 6))
sns.countplot(data=concatenated_df, x='WorkstationName_video')
plt.xlabel("WorkstationName_vi, fontsize=16deo")
plt.ylabel("Count", fontsize=16)
plt.title("Distribution of WorkstationName_video", fontsize=20)
plt.legend(labels=["WorkstationName_video"])
plt.xticks(rotation=45)
save_and_show("distribution_WorkstationName_video.png")

# For columns starting with "AvgPeak" (plot together)
plt.figure(figsize=(8, 6))
sns.kdeplot(concatenated_df['AvgPeakVelocityHRRightward'], label='AvgPeakVelocityHRRightward', shade=True)
sns.kdeplot(concatenated_df['AvgPeakVelocityHRLeftward'], label='AvgPeakVelocityHRLeftward', shade=True)
plt.xlabel("AvgPeak Velocity", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.title("Distribution of AvgPeak Velocities", fontsize=20)
plt.legend()
save_and_show("distribution_AvgPeak.png")

# For columns starting with "AvgAcc" (plot together)
plt.figure(figsize=(8, 6))
sns.kdeplot(concatenated_df['AvgAccuracyHRRightward'], label='AvgAccuracyHRRightward', shade=True)
sns.kdeplot(concatenated_df['AvgAccuracyHRLeftward'], label='AvgAccuracyHRLeftward', shade=True)
plt.xlabel("Avg Accuracy", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.title("Distribution of Avg Accuracies", fontsize=20)
plt.legend()
save_and_show("distribution_AvgAccuracy.png")

# For columns starting with "AvgLatency" (plot together)
plt.figure(figsize=(8, 6))
sns.kdeplot(concatenated_df['AvgLatencyHRRightward'], label='AvgLatencyHRRightward', shade=True)
sns.kdeplot(concatenated_df['AvgLatencyHRLeftward'], label='AvgLatencyHRLeftward', shade=True)
plt.xlabel("Avg Latency", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.title("Distribution of Avg Latencies", fontsize=20)
plt.legend()
save_and_show("distribution_AvgLatency.png")