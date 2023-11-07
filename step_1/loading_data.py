import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/emotion_dataset.csv')

#print(df.head())

#print(df['Emotion'].value_counts())

sns.countplot(x='Emotion', data=df)


# display graph
plt.show()
