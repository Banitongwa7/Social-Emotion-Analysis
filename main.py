import neattext.functions as nfx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# step 1

df = pd.read_csv('data/emotion_dataset.csv')

#print(df.head())
#print(df['Emotion'].value_counts())
#sns.countplot(x='Emotion', data=df)

# step 2
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)


# step 3
x = df['Clean_Text']
y = df['Emotion']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



# display graph
#plt.show()
