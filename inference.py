#!/usr/bin/python3
# inference.py

#Import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data collection and Processing

# In[2]:


#Loading csv data to pandas DataFrame
titanic_data = pd.read_csv('https://mphackathon.blob.core.windows.net/mhackathaon/train.csv')


# In[3]:


#printing the first 5 rows of the dataframe
titanic_data.head()


# In[4]:


#Number of rows and columns
titanic_data.shape


# In[5]:


#Getting some information about the3 data
titanic_data.info()


# In[6]:


#checking the number of missing values
titanic_data.isnull().sum()


# Handling Some missing values
# 

# In[7]:


#Droping the 'Cabin' column on the dataframe
titanic_data = titanic_data.drop(columns='Cabin',axis=1)


# In[8]:


#Replacing the missing values in 'Age' column with the mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[9]:


#Finding the mode value of 'Embarked' column
print(titanic_data['Embarked'].mode())


# In[10]:


print(titanic_data['Embarked'].mode()[0])


# In[11]:


titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[12]:


#checking the number of missing values
titanic_data.isnull().sum()


# In[13]:


#Gettig some statistical masures about the data
titanic_data.describe()


# In[14]:


#Fiding the number of people who survived and did not survice
titanic_data['Survived'].value_counts()


# 0 --> People who did not survive </br>
# 1 --> People who srvived

# Data Visualization

# In[15]:


sns.set()


# In[16]:


#Making a countplot for 'Survived' column
sns.countplot('Survived', data=titanic_data)


# In[17]:


titanic_data['Sex'].value_counts()


# In[18]:


#Making a countplot for 'Sex' column
sns.countplot('Sex', data=titanic_data)


# In[19]:


#Number of Suvivers gender wise
sns.countplot('Sex', hue='Survived', data=titanic_data)


# In[20]:


#Making a countplot for 'Pclass' column
sns.countplot('Pclass', data=titanic_data)


# In[21]:


#Number of Suvivers gender wise
sns.countplot('Pclass', hue='Survived', data=titanic_data)


# Encoding the categorical column

# In[22]:


titanic_data['Sex'].value_counts()


# In[23]:


titanic_data['Embarked'].value_counts()


# In[24]:


# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder
 
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(titanic_data['Sex'])
 
# printing label
label


# In[25]:


# removing the column 'Purchased' from df
# as it is of no use now.
titanic_data.drop("Sex", axis=1, inplace=True)
 
# Appending the array to our dataFrame
# with column name 'Purchased'
titanic_data["Sex"] = label
 
# printing Dataframe
titanic_data


# In[26]:


# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder
 
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(titanic_data['Embarked'])
 
# printing label
label


# In[27]:


# removing the column 'Purchased' from df
# as it is of no use now.
titanic_data.drop("Embarked", axis=1, inplace=True)
 
# Appending the array to our dataFrame
# with column name 'Purchased'
titanic_data["Embarked"] = label
 
# printing Dataframe
titanic_data


# In[28]:


titanic_data.head()


# Separating Features and Target

# In[29]:


X = titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived'], axis=1)
Y = titanic_data['Survived']


# In[30]:


print(X)


# In[31]:


print(Y)


# Splitting the data int training data and test data

# In[32]:


X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[33]:


print(X.shape,X_train.shape,X_test.shape)


# Model Training

# In[34]:


model = LogisticRegression()


# In[35]:


#Training the Logistic regression model with training data
model.fit(X_train,Y_train)


# Model Evaluation
#print(model)
# Accuracy Score

# In[36]:


# #accuracy of train data
 X_train_prediction  = model.predict(X_train)


# # In[37]:


#print(X_train_prediction)


# # In[38]:


# training_data_accuracy = accuracy_score(Y_train,X_train_prediction)
print('Accuracy Score of training data: ', training_data_accuracy)


# # In[39]:


# #accuracy of test data
X_test_prediction  = model.predict(X_test)


# # In[40]:


# print(X_test_prediction)


# # In[41]:


test_data_accuracy = accuracy_score(Y_test,X_test_prediction)
print('Accuracy Score of training data: ', test_data_accuracy)
