import numpy as np 
import pandas as pd
import joblib

dataset= pd.read_csv('emails.csv')

import re
import nltk
nltk.download('wordnet')
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
spam_msg=[]
for i in range(5728):
    spam= re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    spam=spam.lower()
    spam= spam.split()
    ps=PorterStemmer()
    all_stopwords= stopwords.words('english')
    spam= [ps.stem(word) for word in spam if not word in all_stopwords]
    spam= ' '.join(spam)
    spam_msg.append(spam)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
x=cv.fit_transform(spam_msg).toarray()
y= dataset.iloc[:,1].values
joblib.dump(cv,'countvectorizer_email.pkl')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.naive_bayes import GaussianNB,BernoulliNB
classifier= GaussianNB()
classifier.fit(x_train,y_train)

classifier_2= BernoulliNB()
classifier_2.fit(x_train,y_train)


# Save the trained model to a file
joblib.dump(classifier_2, 'spam_classifier_model_email.pkl')


y_predict= classifier.predict(x_test)
# print(np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)),1))

y_predict_2= classifier_2.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
cm=confusion_matrix(y_test,y_predict)
print(cm)
print(accuracy_score(y_test,y_predict))
print(accuracy_score(y_test,y_predict_2))
