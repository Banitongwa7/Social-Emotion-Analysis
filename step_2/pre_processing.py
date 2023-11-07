import neattext.functions as nfx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/emotion_dataset.csv')

df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)

print(df)