import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
import textwrap
import contextily as ctx
import matplotlib.pyplot as plt
import seaborn as sns

#_______________________________________________COLAB NOTEBOOK___________________________________________________________#
from google.colab import drive
drive.mount('/content/drive')


import os

# Path to the folder or file in your Google Drive
folder_path = '/content/drive/MyDrive/guvi'  # Folder containing the file

# List files in the specified folder
files = os.listdir(folder_path)
print("Files in folder:", files)


import pandas as pd

# Construct the full file path
file_path = os.path.join(folder_path, 'Crime_Data.xlsx')

# Load the Excel file
df = pd.read_excel(file_path)

# Display the first few rows
print("Data preview:")
print(df.head())


print("Dataset Info:")
print(df.info())

print("Summary Statistics:")
print(df.describe())


missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Creating a DataFrame to display the results
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Missing Percentage': missing_percentage
}).sort_values(by='Missing Values', ascending=False)

# Display the result
print(missing_data)

# Filling numeric missing values with the median
numeric_columns = ['Longitude', 'Latitude', 'X Coordinate', 'Y Coordinate']
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Filling categorical missing values with the mode
if 'Location Description' in df.columns:
    df['Location Description'] = df['Location Description'].fillna(df['Location Description'].mode()[0])

# Filling missing values in 'Location' with 'Unknown'
if 'Location' in df.columns:
    df['Location'] = df['Location'].fillna('Unknown')

# Validate missing values
print("Missing values after handling:\n", df.isnull().sum())
print(df.columns)


# Convert 'Date' column to datetime and extract components
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour
df['Day_of_Week'] = df['Date'].dt.day_name()

# Drop 'Ward' and 'Community Area' if they exist
df.drop(columns=['Ward', 'Community Area'], inplace=True, errors='ignore')

# Standardize categorical data (e.g., 'Primary Type')
df['Primary Type'] = df['Primary Type'].str.strip().str.title()
df['Location Description'] = df['Location Description'].str.strip().str.title()


# Check for remaining missing values
print(df.isnull().sum())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Drop duplicates if necessary
df.drop_duplicates(inplace=True)

# Preview cleaned data
print(df.head())


print(df.describe())
print(df.info())

print(df.describe())  # Statistical summary for numerical columns
print(df['Primary Type'].value_counts())  # Frequency of crime types
print(df['Day_of_Week'].value_counts())  # Frequency of crimes by day of the week


# Define the directory and filename
save_directory = "/content/drive/MyDrive/guvi"
save_filename = "cleaned_crime_data.csv"

# Full file path
cleaned_file_path = f"{save_directory}/{save_filename}"

# Save the cleaned dataset
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved successfully at: {cleaned_file_path}")


#__________________________________________________V S CODE_________________________________________________________#
# Load the dataset
chicago = pd.read_csv(r'C:\Chicago\cleaned_crime_data.csv')

# Convert date columns to datetime format
chicago['Date'] = pd.to_datetime(chicago['Date'])
chicago['Updated On'] = pd.to_datetime(chicago['Updated On'])

# Drop duplicate rows based on the 'ID' column
chicago.drop_duplicates(subset=['ID'], inplace=True)

# Extract temporal features from the 'Date' column
chicago['Month'] = chicago['Date'].dt.month
chicago['Day'] = chicago['Date'].dt.day
chicago['Hour'] = chicago['Date'].dt.hour
chicago['Day_of_Week'] = chicago['Date'].dt.day_name()

# Add a 'Season' column
chicago['Season'] = chicago['Month'].apply(
    lambda x: 'Winter' if x in [12, 1, 2] else 
              'Spring' if x in [3, 4, 5] else 
              'Summer' if x in [6, 7, 8] else 
              'Fall'
)

# Rectify year discrepancies by keeping only valid years
valid_years = range(2001, 2025)
chicago = chicago[chicago['Year'].isin(valid_years)]

# Validate latitude and longitude ranges
lat_min, lat_max = 41.5, 42.2  # Approximate bounds for Chicago
long_min, long_max = -88, -87.5
chicago = chicago[(chicago['Latitude'].between(lat_min, lat_max)) & (chicago['Longitude'].between(long_min, long_max))]

# Check for and remove rows where latitude or longitude is 0
chicago = chicago[(chicago['Latitude'] != 0) & (chicago['Longitude'] != 0)]

# Ensure 'ID' is unique
assert chicago['ID'].is_unique, "ID column still contains duplicates!"

# Add crime hotspot clustering using K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
chicago['Cluster'] = kmeans.fit_predict(chicago[['Latitude', 'Longitude']])

# Analyze arrest rate over time
arrest_rate_by_year = chicago.groupby('Year')['Arrest'].mean()

# Output cleaned dataset information
print(chicago.info())
print(chicago.head())

# Summary statistics for latitude and longitude
print(chicago[['Latitude', 'Longitude']].describe())

# Value counts for key columns
print(chicago['Year'].value_counts().sort_index())
print(chicago['Primary Type'].value_counts())
print(chicago['IUCR'].value_counts().head())
print(chicago['FBI Code'].value_counts().head())

# Check for null values
print(chicago.isnull().sum())

# Export cleaned dataset (if needed)
chicago.to_csv(r'C:\Chicago\refined_crime_data.csv', index=False)

# Print additional insights
print("Arrest Rate by Year:")
print(arrest_rate_by_year)
print("Domestic Crimes by Type:")
domestic_crimes = chicago[chicago['Domestic'] == True]
print(domestic_crimes['Primary Type'].value_counts())

# Display basic information about the dataset
print("Dataset Overview:\n", chicago.head())
print("Columns:\n", chicago.columns)

#_________________________________________VISUALIZATION FOR EDA_______________________________________________#

# --- Temporal Analysis ---
# Extract time-related features from datetime
chicago['Date'] = pd.to_datetime(chicago['Date'])
chicago['Year'] = chicago['Date'].dt.year
chicago['Month'] = chicago['Date'].dt.month
chicago['Day'] = chicago['Date'].dt.day
chicago['Hour'] = chicago['Date'].dt.hour

# Crime Trends Over Time
yearly_trends = chicago.groupby('Year').size()
plt.figure(figsize=(10, 6))
yearly_trends.plot(kind='line', marker='o', title='Crime Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()

# Peak crimes by hour
hourly_crimes = chicago.groupby('Hour').size()

# Automatically wrap labels
wrapped_labels = [textwrap.fill(label, 10) for label in hourly_crimes.index.astype(str)]

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    hourly_crimes.values.reshape(1, -1),
    annot=True,
    fmt='d',
    cmap='coolwarm',
    cbar=False
)

# Set the title and axis labels
plt.title("Crime Frequency by Hour")
plt.xlabel("Hour of Day")

# Use the wrapped labels for the x-axis
plt.xticks(ticks=range(len(hourly_crimes)), labels=wrapped_labels)

# Adjust layout
plt.tight_layout()
plt.show()

# --- Geospatial Analysis ---
if 'Latitude' in chicago.columns and 'Longitude' in chicago.columns:
    # Plot hotspots using a map
    crime_map = folium.Map(location=[chicago['Latitude'].mean(), chicago['Longitude'].mean()], zoom_start=12)
    high_risk_areas = chicago.groupby(['Latitude', 'Longitude']).size()
    high_risk_areas = high_risk_areas[high_risk_areas > high_risk_areas.quantile(0.95)]

    for (lat, lon), count in high_risk_areas.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=min(count / 10, 10),  # Scale the radius
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6
        ).add_to(crime_map)

# --- Geospatial Analysis ---
if 'Latitude' in chicago.columns and 'Longitude' in chicago.columns:
    # Initialize the map centered at the average latitude and longitude
    crime_map = folium.Map(
        location=[chicago['Latitude'].mean(), chicago['Longitude'].mean()],
        zoom_start=12
    )
    
    # Group crimes by latitude and longitude, and identify high-risk areas (top 5%)
    high_risk_areas = chicago.groupby(['Latitude', 'Longitude']).size()
    high_risk_areas = high_risk_areas[high_risk_areas > high_risk_areas.quantile(0.95)]

    # Add circle markers for high-risk areas
    for (lat, lon), count in high_risk_areas.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=min(max(count / 10, 5), 15),  # Dynamically scale radius
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            tooltip=f"Crimes: {count}"  # Add tooltip to display crime count
        ).add_to(crime_map)

    # Add a title-like marker for better context
    folium.map.Marker(
        [chicago['Latitude'].mean(), chicago['Longitude'].mean()],
        icon=folium.DivIcon(html=f"""
            <div style="
                font-size: 16px; 
                font-weight: bold; 
                color: black; 
                text-align: center;">
                High-Risk Crime Areas
            </div>
        """)
    ).add_to(crime_map)

    # Display the map (if in a Jupyter Notebook or interactive environment)
    crime_map


# --- Crime Type Analysis ---
crime_types = chicago['Primary Type'].value_counts()
plt.figure(figsize=(12, 8))
crime_types[:10].plot(kind='bar', color='skyblue')
plt.title('Top 10 Crime Types')
plt.xlabel('Crime Type')
plt.ylabel('Number of Incidents')
plt.show()

# --- Arrest Analysis ---
if 'Arrest' in chicago.columns:
    arrest_rates = chicago.groupby('Primary Type')['Arrest'].mean() * 100
    arrest_rates = arrest_rates.sort_values(ascending=False)

    # Wrapping labels
    wrapped_labels = [textwrap.fill(label, width=10) for label in arrest_rates[:10].index]

    plt.figure(figsize=(12, 8))
    plt.bar(wrapped_labels, arrest_rates[:10], color='lightgreen')
    plt.title('Top 10 Crimes by Arrest Rate')
    plt.xlabel('Crime Type')
    plt.ylabel('Arrest Rate (%)')
    plt.xticks(rotation=45, ha='right')  # Ensure readability for wrapped labels
    plt.tight_layout()
    plt.show()
# --- Severity Analysis ---
if 'Severity' in chicago.columns:
    severity_distribution = chicago['Severity'].value_counts()
    plt.figure(figsize=(10, 6))
    severity_distribution.plot(kind='bar', color='tomato')
    plt.title('Distribution of Crime Severity')
    plt.xlabel('Severity')
    plt.ylabel('Number of Crimes')
    plt.show()

# --- Domestic vs. Non-Domestic Crimes ---
if 'Domestic' in chicago.columns:
    domestic_vs_non_domestic = chicago['Domestic'].value_counts()
    plt.figure(figsize=(8, 5))
    domestic_vs_non_domestic.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Domestic vs. Non-Domestic Crimes')
    plt.xlabel('Crime Type')
    plt.ylabel('Number of Incidents')
    plt.xticks(ticks=[0, 1], labels=['Non-Domestic', 'Domestic'], rotation=0)
    plt.show()

# --- Location-Specific Analysis ---
if 'Location Description' in chicago.columns:
    location_crimes = chicago['Location Description'].value_counts()
    plt.figure(figsize=(12, 8))
    location_crimes[:10].plot(kind='barh', color='teal')
    plt.title('Top 10 Crime Locations')
    plt.xlabel('Number of Incidents')
    plt.ylabel('Location Description')
    plt.show()

# --- Seasonal Analysis ---
chicago['Season'] = chicago['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})
seasonal_trends = chicago.groupby('Season').size()
plt.figure(figsize=(10, 6))
seasonal_trends.plot(kind='bar', color='orange')
plt.title('Crime Distribution by Season')
plt.xlabel('Season')
plt.ylabel('Number of Crimes')
plt.show()

# --- Repeat Offenders and Recidivism Analysis ---
if 'Offender ID' in chicago.columns:
    repeat_offenders = chicago['Offender ID'].value_counts()
    repeat_offenders = repeat_offenders[repeat_offenders > 1]
    print(f"Number of repeat offenders: {len(repeat_offenders)}")

    plt.figure(figsize=(10, 6))
    repeat_offenders.plot(kind='hist', bins=30, color='purple', alpha=0.7)
    plt.title('Frequency of Repeat Offenses')
    plt.xlabel('Number of Crimes by an Offender')
    plt.ylabel('Number of Offenders')
    plt.grid(True)
    plt.show()


    # Scatter plot for high-risk areas
    top_10_areas = high_risk_areas.nlargest(10)
    for (lat, lon), count in top_10_areas.items():
        plt.text(lon, lat, str(count), fontsize=9, ha='center', color='black', 
                 bbox=dict(facecolor='white', alpha=0.7), zorder=6)

    # Add basemap using contextily with OpenStreetMap
    ax.set_title('High-Risk Crime Areas with Map Context', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)

    # Adjust plot limits to focus on data
    ax.set_xlim(high_risk_areas.index.get_level_values(1).min() - 0.01,
                high_risk_areas.index.get_level_values(1).max() + 0.01)
    ax.set_ylim(high_risk_areas.index.get_level_values(0).min() - 0.01,
                high_risk_areas.index.get_level_values(0).max() + 0.01)

    plt.tight_layout()
    plt.show()


