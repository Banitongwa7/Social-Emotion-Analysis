import neattext.functions as nfx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib


# step 1

df = pd.read_csv('data/emotion_dataset.csv')

#print(df.head())
#print(df['Emotion'].value_counts())
#sns.countplot(x='Emotion', data=df)

# step 2
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles) # remove user handles
df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords) # remove stopwords


# step 3
x = df['Clean_Text']
y = df['Emotion']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# step 4 
# training model
pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])
pipe_lr.fit(x_train, y_train)
pipe_lr.score(x_test, y_test)

pipe_svm = Pipeline(steps=[('cv', CountVectorizer()), ('svm', SVC(kernel='rbf', C=10))])
pipe_svm.fit(x_train, y_train)
pipe_svm.score(x_test, y_test)

pipe_rf = Pipeline(steps=[('cv', CountVectorizer()), ('rf', RandomForestClassifier(n_estimators=10))])
pipe_rf.fit(x_train, y_train)
pipe_rf.score(x_test, y_test)

# step 5
# saving model
pipeline_file = open('models/emotion_classifier.pkl', 'wb')
joblib.dump(pipe_lr, pipeline_file)
pipeline_file.close()

# display graph
#plt.show()
