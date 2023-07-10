import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle



df = pd.read_csv('mail_data.csv')

data=df.where((pd.notnull(df)),'')

data.loc[data['Category']=='spam', 'Category',]=0
data.loc[data['Category']=='ham', 'Category',]=1

X=data['Message']
Y=data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

input_your_mail = ["Congratulations!! you have won a car for FREE FREE!!!"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)

if(prediction[0]==1):
    print("Ham mail")
else:
    print("Spam mail")



pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(feature_extraction, open('vectorizer.pkl', 'wb'))