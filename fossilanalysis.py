# Function that returns two dataframes

import pandas as pd

def oil_electricity_production(yourFile):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(yourFile, skiprows=4)
    
    # Extract data for years 2010-2019
    years_as_columns = df.loc[:, 'Country Name':'2019']
    
    # Convert year column names to strings
    years_as_columns.columns = [str(col) if col.isdigit() else col for col in years_as_columns.columns]
    
    # Transpose the DataFrame to get a country-centric view
    countries_as_columns = years_as_columns.transpose()
    
    # Replace missing values with 0
    countries_as_columns.fillna(0, inplace=True)
    
    # Set the column names for the countries DataFrame
    countries_as_columns.columns = countries_as_columns.iloc[0]
    countries_as_columns = countries_as_columns[1:]
    countries_as_columns.index.name = 'Year'
    
    # Rename the 'Country Name' column to 'Year' and set it as the index for the years DataFrame
    years_as_columns = years_as_columns.rename(columns={'Country Name': 'Year'})
    years_as_columns.set_index('Year', inplace=True)
    
    return years_as_columns, countries_as_columns


years,countries = oil_electricity_production('API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5358479.csv')

years

countries

years.info()

countries.info()

countries.describe()

years.describe()

# Correlation heat map for all the 9 countries
import pandas as pd

# define the list of nine countries
countries = ['United States', 'Russian Federation', 'China','Germany', 'Brazil', 'Kenya', 'Mexico', 'India', 'France']

# read in the four CSV files as Pandas dataframes
df1 = pd.read_csv('API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5358479.csv', skiprows=4)
df2 = pd.read_csv('API_EG.ELC.RNWX.KH_DS2_en_csv_v2_5358682.csv', skiprows=4)
df3 = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5358347.csv', skiprows=4)
df4 = pd.read_csv('c4276632-5707-4884-a485-d1a0cd46aaa7_Data.csv', skiprows=0)

# extract only the 'Country Name' and '2015' columns for the selected countries
df1 = df1[df1['Country Name'].isin(countries)][['Country Name', '2015']]
df2 = df2[df2['Country Name'].isin(countries)][['Country Name', '2015']]
df3 = df3[df3['Country Name'].isin(countries)][['Country Name', '2015']]
df4 = df4[df4['Country Name'].isin(countries)][['Country Name', '2015']]

# merge the four dataframes into a single dataframe using the 'Country Name' column as the key
merged_df = pd.merge(df1, df2, on='Country Name', suffixes=('_1', '_2'))
merged_df = pd.merge(merged_df, df3, on='Country Name')
merged_df = pd.merge(merged_df, df4, on='Country Name')

# rename the '2015' columns to match the corresponding source dataframes
merged_df.columns = ['Country Name', 'NonRenewableElectricity', 'RenewableElectricity', 'CO2 Emissions(kt)','GDP']

# print the final merged dataframe
merged_df

merged_df.info()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in the CSV file as a Pandas dataframe
df = merged_df

# remove the first column ('Country Name') since we don't want to plot it
df = df.iloc[:, 1:]

# clean the 'GDP' column by removing any non-numeric characters
df['GDP'] = df['GDP'].str.replace(',', '').str.replace('$', '').astype(float)

# calculate the correlation matrix
corr_matrix = df.corr()

# plot the correlation matrix as a heatmap with correlation values displayed inside each box
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap='coolwarm')
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)

# loop over each box and display the correlation value as text inside the box
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2), ha='center', va='center', color='w')

plt.colorbar(im)
plt.show()

# Correlation analysis for the United States
import pandas as pd

# set the name of the country you want to select
country_name = "United States"

# set the filenames for the four CSV files
filename1 = "API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5358479.csv"
filename2 = "API_EG.ELC.RNWX.KH_DS2_en_csv_v2_5358682.csv"
filename3 = "API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5358347.csv"
filename4 = "c4276632-5707-4884-a485-d1a0cd46aaa7_Data.csv"

# load the data for the first three CSV files
df1 = pd.read_csv(filename1, skiprows=4, usecols=['Country Name', '2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006'])
df2 = pd.read_csv(filename2, skiprows=4, usecols=['Country Name', '2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006'])
df3 = pd.read_csv(filename3, skiprows=4, usecols=['Country Name', '2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006'])

# concatenate the data from the first three CSV files
df_combined = pd.concat([df1, df2, df3], ignore_index=True)

# load the data for the fourth CSV file
df4 = pd.read_csv(filename4, header=0, usecols=['Country Name', '2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006'])

# concatenate the data from the fourth CSV file
df_combined = pd.concat([df_combined, df4], ignore_index=True)

# select the rows for the chosen country and years
df_selected = df_combined.loc[df_combined['Country Name'] == country_name, ['2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006']]

# set the row index to be the years
df_selected.index = ['NonRenewableElectricity', 'RenewableElectricity', 'CO2 Emissions(kt)','GDP']

# print the resulting dataframe
df_selected

df_transposed = df_selected.transpose()

# Set the column names
df_transposed.columns = ['NonRenewableElectricity', 'RenewableElectricity', 'CO2 Emissions(kt)','GDP']

# Reset the index to make the years as rows
df_transposed = df_transposed.reset_index()

# Rename the 'index' column to 'Year'
df_transposed = df_transposed.rename(columns={'index': 'Year'})

df_transposed = df_transposed.apply(pd.to_numeric, errors='coerce')

df_transposed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in the CSV file as a Pandas dataframe
df = df_transposed

# remove the first column ('Year') since we don't want to plot it
df = df.iloc[:, 1:]

# calculate the correlation matrix
corr_matrix = df.corr()

# plot the correlation matrix as a heatmap with correlation values displayed inside each box
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap='coolwarm')
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)

# loop over each box and display the correlation value as text inside the box
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2), ha='center', va='center', color='w')

plt.colorbar(im)
plt.show()



# Corelation Analysis for Brazil
import pandas as pd

# set the name of the country you want to select
country_name = "Brazil"

# set the filenames for the four CSV files
filename1 = "API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5358479.csv"
filename2 = "API_EG.ELC.RNWX.KH_DS2_en_csv_v2_5358682.csv"
filename3 = "API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5358347.csv"
filename4 = "c4276632-5707-4884-a485-d1a0cd46aaa7_Data.csv"

# load the data for the first three CSV files
df1 = pd.read_csv(filename1, skiprows=4, usecols=['Country Name', '2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006'])
df2 = pd.read_csv(filename2, skiprows=4, usecols=['Country Name', '2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006'])
df3 = pd.read_csv(filename3, skiprows=4, usecols=['Country Name', '2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006'])

# concatenate the data from the first three CSV files
df_combined = pd.concat([df1, df2, df3], ignore_index=True)

# load the data for the fourth CSV file
df4 = pd.read_csv(filename4, header=0, usecols=['Country Name', '2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006'])

# concatenate the data from the fourth CSV file
df_combined = pd.concat([df_combined, df4], ignore_index=True)

# select the rows for the chosen country and years
df_selected = df_combined.loc[df_combined['Country Name'] == country_name, ['2015','2014','2013', '2012','2011', '2010','2009','2008','2007','2006']]

# set the row index to be the years
df_selected.index = ['NonRenewableElectricity', 'RenewableElectricity', 'CO2 Emissions(kt)','GDP']

# print the resulting dataframe
df_selected

df_transposedd = df_selected.transpose()

# Set the column names
df_transposedd.columns = ['NonRenewableElectricity', 'RenewableElectricity', 'CO2 Emissions(kt)','GDP']

# Reset the index to make the years as rows
df_transposedd = df_transposedd.reset_index()

# Rename the 'index' column to 'Year'
df_transposedd = df_transposedd.rename(columns={'index': 'Year'})

df_transposedd = df_transposedd.apply(pd.to_numeric, errors='coerce')

df_transposedd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in the CSV file as a Pandas dataframe
df = df_transposedd

# remove the first column ('Year') since we don't want to plot it
df = df.iloc[:, 1:]

# calculate the correlation matrix
corr_matrix = df.corr()

# plot the correlation matrix as a heatmap with correlation values displayed inside each box
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap='coolwarm')
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)

# loop over each box and display the correlation value as text inside the box
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2), ha='center', va='center', color='w')

plt.colorbar(im)
plt.show()

df_transposed.info()

# Grouped Bar Graph
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the csv file
df = pd.read_csv('API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5358479.csv', skiprows=4)

# Select the data for the desired countries and years
countries = ['United States', 'Russian Federation', 'China','Germany', 'Brazil', 'Kenya', 'Mexico', 'India', 'France']
years = [str(year) for year in range(2006, 2016)]
df_selected = df[df['Country Name'].isin(countries)][['Country Name'] + years]

# Set the country name as the index
df_selected.set_index('Country Name', inplace=True)

# Create a grouped bar chart
ax = df_selected.plot(kind='bar', width=0.8, figsize=(12, 8), fontsize=12)

# Set the chart title and axis labels
ax.set_title('Electricity production from oil, gas and coal sources (% of total)', fontsize=16)
ax.set_xlabel('Country', fontsize=14)
ax.set_ylabel('% of electricity production', fontsize=14)

# Set the legend
ax.legend(fontsize=12)

# Show the plot
plt.show()


# MULTIPLE LINE GRAPH
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5358479.csv", skiprows=4)

# Select the countries and years of interest
countries = ['United States', 'Russian Federation', 'China','Germany', 'Brazil', 'Kenya', 'Mexico', 'India', 'France']
years = [str(year) for year in range(2006, 2016)]
df_selected = df[df["Country Name"].isin(countries)][["Country Name"] + years]

# Set the country names as the index
df_selected.set_index("Country Name", inplace=True)

# Create the line chart
fig, ax = plt.subplots(figsize=(12, 8))
for country in countries:
    ax.plot(years, df_selected.loc[country], label=country)
    
# Set the chart title and axis labels
ax.set_title("Electricity Production from Oil, Gas and Coal Sources (% of Total)", fontsize=16)
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("% of Total Electricity Production", fontsize=14)

# Set the legend
ax.legend(fontsize=12)

# Show the chart
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5358479.csv', skiprows=4)

# Select the columns for the 9 countries and the 10 years
countries = ['United States', 'Russian Federation', 'China','Germany', 'Brazil', 'Kenya', 'Mexico', 'India', 'France']
years = [str(year) for year in range(2006, 2016)]
df_selected = df.loc[df['Country Name'].isin(countries), ['Country Name'] + years]

# Pivot the DataFrame so that the years become the index and the countries become the columns
df_pivot = df_selected.set_index('Country Name').T

# Convert the data type from string to float
df_pivot = df_pivot.astype(float)

# Create an area chart
ax = df_pivot.plot(kind='area', stacked=True, alpha=0.7, figsize=(12, 8))

# Set the chart title and axis labels
ax.set_title('Electricity Production from Fossil Fuels (% of total)', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('% of Total', fontsize=12)

# Set the legend outside the chart area
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# Show the chart
plt.show()
