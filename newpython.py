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

# Plot the distribution of accidents by day of the week
plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
sns.countplot(data=data_cleaned, x='day_of_week', palette='Set2')
plt.title('Number of Accidents by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
# Select only numeric columns
numeric_data = data_cleaned.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(data=data_cleaned, x='Borough', palette='magma')
plt.title('Accidents by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()
# Map 0 to 'Non-Fatal', 1 to 'Fatal'
fatal_counts = data_cleaned['Fatality'].map({1: 'Fatal', 0: 'Non-Fatal'}).value_counts()

labels = fatal_counts.index
sizes = fatal_counts.values
colors = ['#FF6F61', '#6EC6FF']
explode = [0.08 if label == 'Fatal' else 0 for label in labels]

plt.figure(figsize=(5, 5))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    startangle=140,
    shadow=False,
    textprops=dict(color="black", fontsize=10)
)

plt.title("Fatal vs Non-Fatal Accidents", fontsize=12)
plt.axis('equal')
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))

