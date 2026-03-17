

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("venv/data/student_raw_data.csv")
print(df.head())
print('\nAll Columns in Dataset:')
print(df.columns)
columns_to_remove=['school','address','famsize','Pstatus',
                   'Medu','Fedu','Mjob',
                   'Fjob','reason','guardian',
                   'paid','activities','nursery','higher',
                   'internet','romantic','Dalc','Walc']
df=df.drop(columns=columns_to_remove)
print("Columns removed successfully")
print(df.columns)
df = df.drop(columns=['traveltime','schoolsup','famsup'])
print("\nFinal Columns:")
print(df.columns)
# add std_id
df.insert(0,'student_id',range(1,len(df)+1))
print(df.head())
# rename a few col
df=df.rename(columns={
    'studytime':'study_time','famrel':'family_relationship',
    'freetime':'free_time','G1':'exam1_score',
    'G2':'exam2_score','G3':'exam3_score'})
print("\n Updated columns.")
print(df.columns)
df=df.reset_index(drop=True)
print(df.head())
# missing value
print("\nMIssing values in each column:")
print(df.isnull().sum())
# clean and save the data
df.to_csv("student_cleaned_data.csv",index=False)