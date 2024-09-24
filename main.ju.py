# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %%
import pandas as pd
import numpy as np
import seaborn as sns # For KDE 
import matplotlib.pyplot as plt # For general plots

# %%
import warnings
warnings.filterwarnings("ignore") # Removing UserWarnings.

# %% [markdown]
"""
## Dataset Loading
"""

# %%
trainData = pd.read_csv("./dataset/train.csv")
testData = pd.read_csv("./dataset/test.csv")

# %% [markdown]
"""
## Exploratory Data Analysis
"""

# %%
trainData.head(10)

# %%
missing_values = trainData.isnull().sum()
print("Missing values in each column:\n", missing_values)

# %% [markdown]
"""
The data has no missing values.
"""

# %%
data_types = trainData.dtypes
print("Data types of columns:\n", data_types)

# %% [markdown]
"""
## Data Visualization
"""

# %%
numBins = 500
figureSize = (8,6)

# %% [markdown]
"""
Since time of travelling is very important, we can divide the day into 4 segments of 6 hours each and then 
analyze the distribution of trip durations in each segment. Along with stuff like the mean and median, as 
well as the geographic distribution of the locations.
"""

# %%

trainData['pickup_datetime'] = pd.to_datetime(trainData['pickup_datetime'])
trainData['dropoff_datetime'] = pd.to_datetime(trainData['dropoff_datetime'])

def getTimeZone(time):
    hour = time.hour
    if hour > 6 and hour <= 12:
        return "Morning"
    elif hour > 12 and hour <= 18:
        return "Afternoon"
    elif hour > 18 and hour <= 24:
        return "Evening"
    else:
        return "Night"

# Storing the new column
trainData['time_zone'] = trainData['pickup_datetime'].apply(getTimeZone)


# %% [markdown]
"""
### Trip Duration Distribution

"""

# %% [markdown]
"""
## 1. Histogram of Trip Duration along with KDE
**Justification**: A histogram with a kernel density estimate (KDE) is effective for visualizing the distribution.

**Insights**: This graph may reveal whether most trips are short or long, indicating common trip patterns.

> __Note__ : Since the simple plot with time duration is too constricted to be of any use, I have simply 
plotted the log-transformed distribution totally and in each time zone.
"""

# %%
plt.figure(figsize=figureSize)
sns.histplot(np.log(trainData['trip_duration']), bins=numBins, color='tan', kde=True)
plt.title('Distribution of Trip Duration')
plt.xlabel('Trip Duration (seconds)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# %%
# Time zone wise statistics.
mean_time_zone = trainData.groupby('time_zone')['trip_duration'].mean()
print(mean_time_zone)
median_time_zone = trainData.groupby('time_zone')['trip_duration'].median()
print(median_time_zone)

# %% [markdown]
"""
### Grouping the data 
Since there is a lot of variation in the trip duration, we can group the data into different time zones.
These help in visualizing the distribution of trip durations in different time zones. Which enables one to 
understand the common trip patterns, traffic patterns, etc.
"""

# %% [markdown]
"""
## Working In Each Time Zone
"""

# %%
trainData['log_trip_duration'] = np.log(trainData['trip_duration'])

fig, axs = plt.subplots(2, 2, figsize=figureSize)

colors = ['darkcyan', 'orange', 'purple', 'green']
time_zones = ['Morning', 'Afternoon', 'Evening', 'Night']
axs_flat = axs.flatten()  

# Plot histograms for each time zone
for ax, color, zone in zip(axs_flat, colors, time_zones):
    sns.histplot(data = trainData[trainData['time_zone'] == zone], x='log_trip_duration',
                 bins=numBins, color=color, kde=True, ax=ax, edgecolor='none')
    ax.set_title(f'Distribution of Trip Duration - {zone}')
    ax.set_xlabel('Trip Duration (seconds)')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.75)

plt.tight_layout()
plt.show()


# %% [markdown]
"""
## 2. Scatter Plot of Pickup vs Dropoff Locations using Seaborn
__Justification__ : A scatter plot is effective for visualizing the geographical distribution of pickup and dropoff locations.
It can also help in finding the prominent hotspots and how they are distributed geographically.

__Insights__ : This graph can reveal patterns in trip origins and destinations across NYC. If plotted across different times
in a day, it can also show potential traffic patterns, as well as the distribution of trips across different neighborhoods.

> __Note__ : We do it again in groups corresponding to the various time zones.
"""

# %%
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Define time zones for plotting
time_zones = ['Morning', 'Afternoon', 'Evening', 'Night']
colors = ['green', 'red']

# Plotting scatter plots for each time zone
for ax, zone in zip(axs.flatten(), time_zones):
    # Filter data for the current time zone
    zone_data = trainData[trainData['time_zone'] == zone]

    # Scatter plot for pickup locations
    sns.scatterplot(x='pickup_longitude', y='pickup_latitude', data=zone_data,
                    alpha=0.5, label='Pickup Locations', color=colors[0], ax=ax)

    # Scatter plot for dropoff locations
    sns.scatterplot(x='dropoff_longitude', y='dropoff_latitude', data=zone_data,
                    alpha=0.5, label='Dropoff Locations', color=colors[1], ax=ax)

    # Set titles and labels for each subplot
    ax.set_title(f'Pickup and Dropoff Locations - {zone}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# %% [markdown]
"""
Clearly there is concenterated traffic in different areas in different time zones. This helps in 
visualizing the geographic distribution of the hot spots, across the time zones.
"""

# %% [markdown]
"""
### Using Heatmap of Pickup and Dropoff Locations
__Justification__ : Unlike traditional scatter plots that may suffer from overplotting in dense areas, 3D surfaces effectively 
summarize the density of points into a continuous representation, making it easier to see trends without losing detail.

__Insights__ : This graph can reveal patterns in trip origins and destinations across NYC. If plotted across different times
in a day, it can also show potential traffic patterns, as well as the distribution of trips across different neighborhoods. In 
the obtained plots, we can see the clear difference between the smoothness of geographical distribution in different time zones.

Morning and Afternoon are generally busy and nights are the least busy time zones. Also, the geographical density corresponding
to these morning and afternoon might be due to office hours, whereas for night it might be because of Emergency Services such as
hospitals.
"""

# %%
pickup_counts = trainData.groupby(['pickup_longitude', 'pickup_latitude']).size().reset_index(name='frequency')
dropoff_counts = trainData.groupby(['dropoff_longitude', 'dropoff_latitude']).size().reset_index(name='frequency')

# %%
from scipy.interpolate import griddata # For 3D interpolation

fig = plt.figure(figsize=(12, 14))

# Define time zones for plotting
time_zones = ['Morning', 'Afternoon', 'Evening', 'Night']

for i, zone in enumerate(time_zones):
    # Filter data for the current time zone
    zone_data_pickup = trainData[trainData['time_zone'] == zone]
    zone_data_dropoff = trainData[trainData['time_zone'] == zone]

    # Group by pickup coordinates and count frequencies
    pickup_counts = zone_data_pickup.groupby(['pickup_longitude', 'pickup_latitude']).size().reset_index(name='frequency')
    dropoff_counts = zone_data_dropoff.groupby(['dropoff_longitude', 'dropoff_latitude']).size().reset_index(name='frequency')

    # Create grid points for interpolation
    grid_lon = np.linspace(pickup_counts['pickup_longitude'].min(), pickup_counts['pickup_longitude'].max(), 100)
    grid_lat = np.linspace(pickup_counts['pickup_latitude'].min(), pickup_counts['pickup_latitude'].max(), 100)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    # Interpolate pickup frequencies onto the grid
    grid_freq_pickup = griddata((pickup_counts['pickup_longitude'], pickup_counts['pickup_latitude']),
                                 pickup_counts['frequency'], (grid_lon, grid_lat), method='cubic')

    # Interpolate dropoff frequencies onto the grid
    grid_freq_dropoff = griddata((dropoff_counts['dropoff_longitude'], dropoff_counts['dropoff_latitude']),
                                  dropoff_counts['frequency'], (grid_lon, grid_lat), method='cubic')

    # Plotting Pickup Frequency Surface for the current time zone
    ax1 = fig.add_subplot(4, 2, i*2 + 1, projection='3d')  # Pickup plots in odd indices (1, 3, 5, 7)
    ax1.plot_surface(grid_lon, grid_lat, grid_freq_pickup, cmap='Blues', alpha=0.7)
    ax1.set_title(f'Pickup Locations Frequency - {zone}')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Plotting Dropoff Frequency Surface for the current time zone
    ax2 = fig.add_subplot(4, 2, i*2 + 2, projection='3d')  # Dropoff plots in even indices (2, 4, 6, 8)
    ax2.plot_surface(grid_lon, grid_lat, grid_freq_dropoff, cmap='Reds', alpha=0.5)
    ax2.set_title(f'Dropoff Locations Frequency - {zone}')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

plt.tight_layout()
plt.show()

# The project can be viewed on my Github as well. : https://github.com/devilunraveled/Kaggle_NYTaxiTripDuration
