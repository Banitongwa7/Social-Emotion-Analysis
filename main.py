import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('data/emotion_dataset.csv')

#print(df.head())

#print(df['Emotion'].value_counts())

print(sns.countplot(x='Emotion', data=df))
