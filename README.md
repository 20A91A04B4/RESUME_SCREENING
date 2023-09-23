# RESUME_SCREENING
#importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#1.Gathering data
import pandas as pd
path="/content/drive/MyDrive/deeplearning/UpdatedResumeDataSet.csv"
a=pd.read_csv(path)
#2.Data preparation
a.info()
a.shape
a.index
a.columns
a.describe
a.isna().sum()
#adding column to dataset
a['clean resume']=''
a.shape
a.head(5)
print(a['Category'])
q=(a['Category'].unique())
print(q)
print(len(q))
print(a['Category'].value_counts())
#3.Data analyse
#plotting of data 
import seaborn as sns
plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
ax=sns.countplot(x="Category", data=a)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.grid()
#function to clean data
import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
    
a['clean resume'] =a.Resume.apply(lambda x: cleanResume(x))

print(a['clean resume'])
a.head()
data=a.copy()
print(data.shape)
#4.Data wrangling
#encoding the values
from sklearn.preprocessing import LabelEncoder
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

data.head()
data['Category'].value_counts()
data.isna()
#5.Train model
#vectorizing the data-it used for converting sentence to vector
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

requiredText = data['clean resume'].values
requiredTarget = data['Category'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,
                                                 shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)

#KNN
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))

----------------------------------------------------------------------------------------------> TESTING
#6.Testing the data

import re

def clResume(resumeText):
    # Remove URLs
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    
    # Remove RT and cc
    resumeText = re.sub('RT|cc', ' ', resumeText)
    
    # Remove hashtags
    resumeText = re.sub('#\S+', '', resumeText)
    
    # Remove mentions
    resumeText = re.sub('@\S+', '  ', resumeText)
    
    # Remove punctuations (except for special characters like ., !, ?)
    resumeText = re.sub('[^A-Za-z0-9.,!?]+', ' ', resumeText)
    
    # Remove non-ASCII characters
    resumeText = re.sub(r'[^\x00-\x7F]+', ' ', resumeText)
    
    # Remove extra whitespace
    resumeText = re.sub('\s+', ' ', resumeText).strip()
    
    return resumeText
# Assuming you have new resume data in a variable called 'new_resumes'
new_resumes = [
    "Experienced@ data analyst with $skills in Python and SQL.",
    "Graphic designer specializing in digital# art and illustration.",
    "Customer service representative with excellent communication skills.",
]

# Clean and preprocess the new data
cleaned_new_resumes = [clResume(resume) for resume in new_resumes]

print(cleaned_new_resumes)
# Transform the cleaned new data using the same TF-IDF vectorizer
from sklearn.feature_extraction.text import CountVectorizer

new_resume_features = word_vectorizer.transform(cleaned_new_resumes)

# Use the trained classifier to make predictions on the new data
new_predictions = clf.predict(new_resume_features)

# Map the predicted labels back to their original categories using LabelEncoder
predicted_categories = le.inverse_transform(new_predictions)
print("1",new_resume_features)
print("2",new_predictions)
print("3",predicted_categories)
# Print the predicted categories for the new resumes
for resume, category in zip(new_resumes, predicted_categories):
    print(f"Resume: {resume}")
    print(f"Predicted Category: {category}")
    print()
