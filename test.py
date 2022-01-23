import pandas as pd

data_raw = pd.read_csv('student-por.csv', sep=';')
data = data_raw.copy()
pd.options.display.max_columns = 150


print(data)

print('Number of participants: ', len(data))
data.head()

print('Is there any missing value? ', data.isnull().values.any())
print('How many missing values? ', data.isnull().values.sum())
data.dropna(inplace=True)
print('Number of participants after eliminating missing values: ', len(data))