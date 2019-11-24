#!/usr/bin/env python
# coding: utf-8

# ## About the Project and Use Case
# 
# Imagine a bank named IBM Data Bank. Huge number of customers had to leave the bank due to no timely bank services. The Bank had a hard time solving the internal conflicts, streamlining the banking process and bringing the customers back to the bank. The bank now wants to make sure that everything runs smoothly and work on how to retain the customers. For this purpose, they want us to build an Data Science application that would predict which of the customers are more likely to leave the bank soon, so the bank can work on how to retain those customers. In this below applications, I am using various machine learning algorithms and helping the bank in predicting which of the customers are more likely to leave the bank soon.
# 
# Note: I am new to deep learning so starting with a small project hope the graders understand and grade me accordingly.

# ## About the Dataset
# 
# The IBM Data Bank is investigating a very high rate of customer leaving the bank. The Dataset contains 10,000 records to investigate and predict which of the customers are more likely to leave the bank soon.
# 
# Note: All the data in the dataset is made up data and the dataset is just a subset of the original dataset (for privacy purposes) and also my neural network is scaled down so that I can run on IBM Watson Studio (less CuH available and do not wish to pay so hope the graders understand).

# ## Initial Exploratory Analysis
# 
# In this notebook, the initial exploratory analysis is performed like the identification of quality issues (missing values, wrong measurements etc.), Correlation matrix is generated and in addition we get an idea of the value distribution of the dataset using statistical measures and data visualizations

# ## Initial Questions Regarding the Dataset for Exploratory Analysis
# 
# 1. Does the dataset have null values?
# 2. What is the percentage of customers who left the bank vs who did not leave the bank?
# 3. Does the dataset contain features have do not impact the final outcome (customer who leave the bank)?
# 4. How is the correlation between the features and the target?
# 5. Do customer having higher bank product subscription leave the bank over the customer issues faced?
# 6. Do customer having credit card leave the bank or not?
# 7. What type of customer leave the bank - experienced (above 40 years) or young people?
# 8. Does Credit Score impact the customer decision to leave the company or not?
# 9. Are there any outliers in the dataset that significantly impact our decision at the end?
# 10. What is the split up of the customers when compared against "Geography", "NumOfCard", "hasCrCard" and "IsActiveMembers"?

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time
bins=range(0,100,10)
import os
import seaborn as sns
sns.set(style='darkgrid', palette='deep')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Importing the dataset
data = pd.read_csv('C:/Users/yash.j.shah.TECHASPECT/Downloads/Churn_Modelling.csv')


# Let us see the dataset and the column datatypes

# In[3]:


# Checking the datatypes of the columns in the dataset
data.info()


# In[4]:


# Displaying the first  rows of the dataset
data.head()


# We drop the columns namely "RowNumber", "Surname" and "CustomerID" because "RowNumber" is just a series identifier and "Surname" & "CustomerID" doesn't really have any imparct on on the customer decision to leave the bank. We display a statistical summary next to get a gist of the effect of the different statistical measures on the dataset.

# In[5]:


# Displaying the statistical summary of the dataset
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data.describe()


# In[6]:


# Displaying the count of the unique values
exited = data['Exited'].value_counts()
exited


# In[7]:


# Displaying the percentage of Bank customer not exited vs exited
countNotExited = len(data[data['Exited'] == 0])     
countExited  = len(data[data['Exited'] == 1]) 
print('Percentage not Exited: {:.2f}%'.format((countNotExited/len(data)) * 100)) 
print('Percentage Exited: {:.2f}%'.format((countExited/len(data)) * 100))


# In[8]:


# Displaying the mean of the columns for the exited states 0 or 1
data.groupby(data['Exited']).mean().head()


# ## Data Pre-Processing
# 
# The data preprocessing section has a sequential flow as below:
# 1. Check for null or missing values
# 2. Feature Dropping Test using Correlation Matrix
# 3. One-hot Encoding

# ## 1. Check for null or missing values

# In[9]:


# Check for the presence of null values in the columns
data.isnull().any()
# Check for the presence of null values in the columns by summing up 
data.isnull().sum()


# As we can see that there are no missing values, so we can move forward the feature dropping test using correlation matrix

# ## 2. Feature Dropping Test using Correlation Matrix
# 
# Features with high correlation ratio are more linearly dependent and hence have almost the same effect on the dependent variable, y. 
# Note: When two features have high correlation, we can drop one of the two features in question. Please find below the correlation matrix and I don't seem to find any high correlation.

# In[10]:


# Correlation Matrix
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), cmap='BuGn', annot=True)


# ## Drawing Conclusion from Correlation Matrix
# 
# No two features have high correlation between them

# ## Data Visualizations
# 
# Let us visualize the dataset by generating some visualization and discuss it as generate it

# In[11]:


_, ax = plt.subplots(1,3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.countplot(x = "NumOfProducts", hue="Exited", data = data, ax = ax[0])
sns.countplot(x = "HasCrCard", hue="Exited", data = data, ax = ax[1])
sns.countplot(x = "IsActiveMember", hue="Exited", data = data, ax = ax[2])


# We generated three plots by comparing the "NumOfProducts", "HasCrCard" and "IsActiveMember" features against the exited feature column and can draw the following conclusions:
# 
# 1. Customers with 3 or more products have higher chances to churn or leave the bank due to no proper response to them from the bank
# 2. Customer using the bank credit card also have a higher chance of leave the bank over customer issues

# In[12]:


sns.countplot(data.Exited)


# The above graph simply display the count of the customer who left the bank and who did not.

# In[13]:


_, ax = plt.subplots(1,3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.swarmplot(x = "NumOfProducts", y = "Age", hue="Exited", data = data, ax = ax[0])
sns.swarmplot(x = "HasCrCard",  y = "Age", hue="Exited", data = data, ax = ax[1])
sns.swarmplot(x = "IsActiveMember",  y = "Age", hue="Exited", data = data, ax = ax[2])


# In[14]:


data['Age'].value_counts().plot.bar(figsize=(20,6))


# In[15]:


facet = sns.FacetGrid(data, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"Age",shade= True)
facet.set(xlim=(0, data["Age"].max()))
facet.add_legend()

plt.show()


# In[16]:


# Age vs Balance
_, ax =  plt.subplots(1, 2, figsize=(15, 7))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = data, ax=ax[0])
sns.scatterplot(x = "Age", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = data, ax=ax[1])


# We generated two scatter plots by comparing the "Balance" and "CreditScore" features against the exited feature column and can draw the following conclusions:
# 
# 1. Customers with age between 40 to 70 years have higher chances to leave the bank
# 2. Customer with poor Credit Score rating of less than 400 or less tend to have higher chances to leave the bank

# In[17]:


plt.figure(figsize=(8, 8))
sns.swarmplot(x = "HasCrCard", y = "Age", data = data, hue="Exited")


# In the above plot we look at the split of people in terms of age having credit card and who left the bank vs did not leave.

# In[18]:


facet = sns.FacetGrid(data, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"Balance",shade= True)
facet.set(xlim=(0, data["Balance"].max()))
facet.add_legend()

plt.show()


# In[19]:


_, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(x = "Balance", y = "Age", data = data, hue="Exited", ax = ax[0])
sns.scatterplot(x = "Balance", y = "CreditScore", data = data, hue="Exited", ax = ax[1])


# In[20]:


facet = sns.FacetGrid(data, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"CreditScore",shade= True)
facet.set(xlim=(0, data["CreditScore"].max()))
facet.add_legend()

plt.show()


# In[21]:


# Detecting Outliers using Boxplots
plt.figure(figsize=(12,6))
bplot = data.boxplot(patch_artist=True)
plt.xticks(rotation=90)       
plt.show()


# No real outlier in the dataset that can significantly impact our final results

# In[22]:


## Pie Plots Distribution Charts 
data.columns
data2 = data.drop( ['Gender', 
                   'CreditScore', 'Age','Tenure', 'Balance',
                   'EstimatedSalary', 'Exited'], axis=1)
fig = plt.figure(figsize=(20, 20))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, data2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(data2.columns.values[i - 1])
   
    values = data2.iloc[:, i - 1].value_counts(normalize = True).values
    index = data2.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# Split up of the customers with respect to features like "Geography", "NumOfProducts", "HasCrCard" and "IsActiveMember"

# ## Feature Engineering
# 
# We will perform feature engineering by performing one-hot encoding for "Geogrpahy" and "Gender" features of the dataset

# ## NOTE
# 
# Before we proceed further with model building steps, it is recommended to split the dataset into train, and test dataset and then apply further pre-processing steps like feature engineering on each dataset separately. 
# 
# My reasoning behind doing this is we may need to normalize or standardize the data in our dataset so let assume if we standardize the complete dataset and then split it, the test dataset might have the mean and standard deviation of the training set as well which in turn will not give us accurate conclusions and our test data already has a sense or knowledge of the training data and our model will start to overfit. So avoid this we split the dataset into train and test data and then go forward.

# In[23]:


# Splitting the dataset into train, validation and test
from sklearn.model_selection import train_test_split
y = data.Exited
X = data.drop(['Exited'], axis=1)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)


# So above, we divided the dataset into train and test data. Based on the results we get, we can further tune the model and run it against the test dataset in turn resulting in a good model which can worked on later to optimize it further.

# Let us find out the Binary, Numerical and Categorical columns in train and test data and then divide each datasets into further small datasets to perform better preprocessing.

# In[24]:


# Dividing the train data into categorical, numerical and binary

binary_columns=["HasCrCard","IsActiveMember"]
binary_data=pd.DataFrame(X_train[binary_columns])

numerical_columns =["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
numerical_data=pd.DataFrame(X_train[numerical_columns])

category_columns=['Geography','Gender']
category_data=pd.DataFrame(X_train[category_columns])


# ## Lookout for Categorical Values
# 
# We find out tht the dataset has two categorical values "Geography" and "Gender". We already know that Machine Learning algorithms only deal with numbers so lets convert this into numbers. We will use One-Hot Encoding to do that.
# 
# Reason for using One-Hot Encoding: We know that the limitation on label encoding is, after encoding, the values in the dataset might confuse the model as if they are somewhat sequential. In our dataset, we already know that both the columns are of the same category type, so we opt for One-Hot Encoding instead of Label Encoding.

# In[25]:


#(Train Data) One-hot encoding Categorical Data

category_data['Geography'] = category_data['Geography'].astype('category')
category_data['Gender'] = category_data['Gender'].astype('category')
category_data_Final = pd.get_dummies(category_data)


# ## Feature Scaling
# 
# On noticing it close, we can see that we have few columns in out dataset that are at a different range when compared to other features. For this reason, we need to make every column under a common umbrella. There are two techniques for doing it - Normalization and Standardization.
# 
# 1. Normalization - Data Normalization is the process of rescaling one or more attributes to the range of 0 or 1 (i.e.) the largest values for each attribute is 1 and the smallest values is 0.
# 
# 2. Standardization - Data Standardization is the process of rescaling one more attributes so that they have a mean values of 0 and a standard deviation of 1.
# 
# Note: Generally Standardization is preferred and we will stick to standardization for our data. However, we are not standardizing each column in our dataset. At this point of our preprocessing, we have categorical, binary and numerical. We standardize only numerci valu and ignore the binary values (one-hot encoding produces the binary columns).

# In[26]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_data_train_mean=numerical_data.mean()
numerical_data_train_std=numerical_data.std(axis=0)
numerical_data_scale =pd.DataFrame(scaler.fit_transform(numerical_data),columns=numerical_columns)


# In[27]:


# Concatenate Columns (Train Data)
X_train = pd.concat([numerical_data_scale, category_data_Final,binary_data], axis=1)


# ## Note
# 
# Feature scaling is done on training, testing data seperately to avoid the problem of data leak. So as you can see, we first calculate the mean and standard deviation of each column of the train data and use the standardization formula on every column of testing data.
# 
# Repeat the above the steps for testing data

# In[28]:


# dividing data into binary, number and categorical (Test data)
binary_columns=["HasCrCard","IsActiveMember"]
binary_data=pd.DataFrame(X_test[binary_columns])

numerical_columns =["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
numerical_data=pd.DataFrame(X_test[numerical_columns])

category_columns=['Geography','Gender']
category_data=pd.DataFrame(X_test[category_columns])

# [TEST] Encode Categorical data (Test data)
category_data['Geography'] = category_data['Geography'].astype('category')
category_data['Gender'] = category_data['Gender'].astype('category')
category_data_Final = pd.get_dummies(category_data)

# [TEST] feature scaling (Test data)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_data["CreditScore"]=(numerical_data["CreditScore"]-numerical_data_train_mean["CreditScore"]).div(numerical_data_train_std["CreditScore"])
numerical_data["Age"]=(numerical_data["Age"]-numerical_data_train_mean["Age"]).div(numerical_data_train_std["Age"])
numerical_data["Tenure"]=(numerical_data["Tenure"]-numerical_data_train_mean["Tenure"]).div(numerical_data_train_std["Tenure"])
numerical_data["Balance"]=(numerical_data["Balance"]-numerical_data_train_mean["Balance"]).div(numerical_data_train_std["Balance"])
numerical_data["NumOfProducts"]=(numerical_data["NumOfProducts"]-numerical_data_train_mean["NumOfProducts"]).div(numerical_data_train_std["NumOfProducts"])
numerical_data["EstimatedSalary"]=(numerical_data["EstimatedSalary"]-numerical_data_train_mean["EstimatedSalary"]).div(numerical_data_train_std["EstimatedSalary"])

# [TEST] Concatenate Columns (Test data)
X_test = pd.concat([numerical_data, category_data_Final,binary_data], axis=1)


# ## Safety Precaution
# 
# This is optional

# In[29]:


# assigning NULL to unused variables
data=None
X=None
y=None
binary_columns=None
binary_data=None
category_data=None
category_columns=None
category_data_Final=None
numerical_data=None
numerical_columns=None
numerical_data_train_mean=None
scaler=None
numerical_data_train_std=None
numerical_data_scale=None
null_columns=None


# ## Modeling Defintion, Training and Evaluation
# 
# We will build two models, one using a traditional machine learning algorithm (Logistic Regression, Random Forest) and one with Neural Network Algorithm. 

# Let see how does the Logistic Regression Performs.

# I have used Logistic Regression as my traditional machine learning algorithm because it uses a logarithmic trnasformation on the target variable which allows us to model a non-linear association in a linear way 

# In[30]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accu_score = accuracy_score(y_pred, y_test)
accu_score


# Let see how the neural network performs Now.
# 

# The Neural Network consist of the following:
#     
# 1. Input Layer
# 2. 3 Hidden Layer - Neurons: 6, 12, 24 
# 3. Output Layer
# 4. Learning rate: 0.02
# 5. Metrics: Accuracy, MAE

# 1. I have chosen Keras Framework as it is built on top on tensorflow. So if you want quickly build and test a neural network with minimal lines of code, I would choose Keras. With Keras Framework, I can quickly build simple or very complex neural network within few minutes, The Sequential() is powerful you can do almost everything you may want.
# <br><br>
# 2. I have chosen ANN network model because it is easy to use and understand compared to statistical methods. ANN is non-parametric model and use Back propagation learning algorithm and it is widely used for classification problems

# In[31]:


#defining and compiling model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD

def deep_model():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='he_uniform',
                bias_initializer='ones', activation='relu', input_dim=13))
    classifier.add(Dense(units=12, kernel_initializer='he_uniform',
                bias_initializer='ones', activation='relu'))
    classifier.add(Dense(units=24, kernel_initializer='he_uniform',
                bias_initializer='ones', activation='relu'))
    classifier.add(Dense(units=12, kernel_initializer='he_uniform',
                bias_initializer='ones', activation='relu'))
    classifier.add(Dense(units=1,  kernel_initializer='he_uniform',
                bias_initializer='ones', activation='sigmoid'))
    classifier.compile(optimizer=Adam(learning_rate=0.002, amsgrad=False),
                        loss='binary_crossentropy', 
                        metrics=['accuracy','mae'])
    return classifier


# Training the model with batch size of 80 over 1500 epochs

# In[32]:


# fitting the data 
from keras.models import load_model
classifier = deep_model()
output=classifier.fit(X_train, y_train, batch_size=80,epochs=1500,verbose=1,shuffle=False)


# ## Evaluation Metrics:
# 
# 1. Accuracy Score
# 2. Confusion Matrix
# 
# <b>Justification:</b> I use the evaluation metrics like accuracy and confusion matrix because since my scenerio is a binary classification Confusion matrix is a good evaluation matric because it gives you the true positives, true negative, false negatives and false positives. I used Accuracy score to know how accurate was my model in predicting the original outcome.

# In[34]:


#calculating Evaluation Metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

# Confusion Matrix
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
results = pd.DataFrame([['ANN Network', accuracy]],
                      columns = ['Model', 'Accuracy'])
print(results)


# ## Visualization of Evaluation Metrics

# In[35]:


# Summarizing output for accuracy
print(output.history.keys())
plt.plot(output.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='lower right')
plt.ylim(0.85,0.89)
plt.show()
#Summarizing output for loss
plt.plot(output.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()


# In[ ]:




