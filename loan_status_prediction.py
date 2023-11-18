import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# loading the dataset to pandas Dataframe
loan_dataset=pd.read_csv('/content/loanpredictiondataset.csv')

type(loan_dataset)

# printing the first 5 rows of the dataframe
loan_dataset.head()

# number of rows and columns
loan_dataset.shape

loan_dataset.describe()

#number of missing values in each column
loan_dataset.isnull().sum()

# dropping the missing values
loan_dataset=loan_dataset.dropna()

# number of missing values in each column
loan_dataset.isnull().sum()

# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

# printing the first 5 rows of the dataframe
loan_dataset.head()

# Dependent column values
loan_dataset['Dependents'].value_counts()

# replacing the value of 3+ to 4
loan_dataset=loan_dataset.replace(to_replace='3+',value=4)

# dependent values
loan_dataset['Dependents'].value_counts()

# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

# material status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)

# convert categorical columns to numeric values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

loan_dataset.head()

# seperating the data and label
X=loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=loan_dataset['Loan_Status']

print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

classifier=svm.SVC(kernel='linear')

#training the support vector machine model
classifier.fit(X_train,Y_train)

"""Model Evaluation"""

#accuracy score on training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print('Accuracy on training data:',training_data_accuracy)

# accuracy score on training data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print('Accuracy on test data:',test_data_accuracy)


import streamlit as st

st.title('Loan Status Prediction')

st.write('This app predicts the Loan Status.')

st.header('Enter the features for prediction:')
with st.form(key='my_form'):
    X1 = st.number_input('Enter your Loan_ID')
    X2 = st.number_input('Enter your Gender')
    X3 = st.number_input('Enter your Married')
    X4 = st.number_input('Enter your Dependents')
    X5 = st.number_input('Enter your Education')
    X6 = st.number_input('Enter your Self_Employed')
    X7 = st.number_input('Enter your ApplicantIncome')
    X8 = st.number_input('Enter your CoapplicantIncome')
    x9 = st.number_input('Enter your LoanAmount')
    x10 =st.number_input('Enter your Loan_Amount_Term')
    x11 =st.number_input('Enter your Credit_History')
    x12 =st.number_input('Enter your Property_Area')
    submit = st.form_submit_button(label='Submit')
if submit:
   s1=[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12]    #predicting the model
#changing data to numpy array
   array=np.asarray(s1)
#reshaping the array
   r1=array.reshape(1,-1)
   s2=s.transform(r1)
   print(s2)
   p1=classifier.predict(s2)
   print(p1)
   if(p1==1):
      st.write("Loan status")
   else:
      st.write("no Loan status")