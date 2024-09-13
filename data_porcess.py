import pandas as pd
import numpy as np

data=pd.read_csv(r'./data/raw_data.csv')


data['TotalCharges'] = data['TotalCharges'].replace(' ',np.nan).astype('float64')
print(data.isnull().sum())
data['Churn'].value_counts(normalize=True).plot.bar()
data.to_csv(r'./data/final_data.csv')

