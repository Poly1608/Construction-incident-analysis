import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('Construction_Related_Incidents.csv')  # replace with the actual filename
data.head(10)
# Display basic info about the data
print(data.info())

# Clean the data: Remove rows with missing values
data_cleaned = data.dropna()

# Check for duplicates and remove them if any
data_cleaned = data_cleaned.drop_duplicates()

# Check for null values
print(data_cleaned.isnull().sum())

# Display the first few rows of the cleaned data
print(data_cleaned.head())
# 1. Get a summary of numerical columns
print(data_cleaned.describe())
# 2. Display unique values for categorical columns (e.g., 'Record Type Description', 'Borough', 'Community Board')
print(data_cleaned['Record Type Description'].unique())  # Replace with the actual categorical column name
# 3. Convert 'Incident Date' to datetime format
data_cleaned['Incident Date'] = pd.to_datetime(data_cleaned['Incident Date'], errors='coerce')  # Handle invalid parsing gracefully
# 4. Extract additional features like day of the week, year, and month from 'Incident Date'
data_cleaned['day_of_week'] = data_cleaned['Incident Date'].dt.dayofweek  # 0=Monday, 6=Sunday
data_cleaned['year'] = data_cleaned['Incident Date'].dt.year
data_cleaned['month'] = data_cleaned['Incident Date'].dt.month

# 5. Check the cleaned data again to ensure changes
print(data_cleaned.head())
data.head(20)
plt.figure(figsize=(18, 6))  # Adjust the overall figure size

# Plot the distribution of accidents by year
plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
sns.countplot(data=data_cleaned, x='year', palette='viridis')
plt.title('Number of Accidents by Year')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)

# Plot the distribution of accidents by month
plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
sns.countplot(data=data_cleaned, x='month', palette='coolwarm')
plt.title('Number of Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
