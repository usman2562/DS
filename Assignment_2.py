#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# In[27]:


files = []

#Looping through all the files and appending the file name to a list
for i in os.listdir(r"C:\Users\keerb\Downloads\Data Science A2\EyeT"):
    files.append(i)
    
#Ignoring first two files because the folder contains 2 pdf and rest all us csv
files = files[2:]


# In[68]:


print('There are total :-',len(files),"CSV files to be merged")


# In[71]:


merged = pd.DataFrame()
count = 0
for i in files:
    data= pd.read_csv("C:/Users/keerb/Downloads/Data Science A2/EyeT/"+i, low_memory=True)
    if count == 0:
        merged = data.copy()
        count = count +1 
    else:
        merged = merged.append(data)
        count = count +1
    print('There are', 502-count, "files left to merge")


# In[72]:


merged


# In[73]:


merged.to_csv('Merged.csv')


# # Data Merging is completed

# In[1]:


import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
import datetime


# In[2]:


score1 = pd.read_csv(r'C:\Users\pc\Desktop\Data Science A2\USMAN\empathy\Questionnaire_datasetIA.csv', encoding= 'unicode_escape')
score2 = pd.read_csv(r'C:\Users\pc\Desktop\Data Science A2\USMAN\empathy\Questionnaire_datasetIB.csv', encoding= 'unicode_escape')


# In[3]:


Merged = pd.read_csv('combineddf.csv')


# In[4]:


Merged.head()


# In[5]:


score1.head(2)


# In[6]:


score1['Participant nr'].unique()


# In[7]:


Merged['Participant name'].unique()


# # Getting Participants number

# In[8]:


#taking the participant's number from each participant name where the number is the last two character.
#By doing this i can merge with quesstionary score to get the empathy score.

last_two = lambda x: x[-2:]
Merged['Participant nr'] = Merged['Participant name'].apply(last_two)
Merged['Participant nr'] = Merged['Participant nr'].astype(int)


# In[9]:


print(Merged.isnull().sum())


# # Handling Null value 

# In[10]:


Merged.isnull().sum().sum()


# In[11]:


import gc
gc.collect()


# In[12]:


#Dropping all the columns which has greater than 25% of null value in the attribute
for i in Merged.columns:
    if Merged[i].isnull().sum()/Merged.shape[0]*100 > 25:
        Merged.drop(i,axis = 1,inplace = True)
        print('Dropped',i,'column due to high percentage of null values')


# In[13]:


Merged.head()


# In[14]:


#Dropping useless columns
Merged.drop({'Unnamed: 0'},axis = 1,inplace = True)


# In[15]:


gc.collect()


# # Handling categorical Variable (usefull categorical variables)

# In[16]:


from sklearn import preprocessing

# Converting categorical column to numerical columns
le = preprocessing.LabelEncoder()

Merged['Sensor'] = le.fit_transform(Merged['Sensor'])
Merged['Validity left'] = le.fit_transform(Merged['Validity left'])
Merged['Validity right'] = le.fit_transform(Merged['Validity right'])
Merged['Eye movement type'] = le.fit_transform(Merged['Eye movement type'])


# In[17]:


Merged.dtypes


# In[18]:


gc.collect()


# # Handling DACSmm attributes

# In[19]:


#Changing , to . so we can convetr DACSmm to mm by dividinmg with 10 because mm is easier to measure and it will normalize the data
string_replace =  ['Eye position left Z (DACSmm)','Eye position right X (DACSmm)','Eye position right Y (DACSmm)',
 'Eye position right Z (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)',
 'Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z',
 'Gaze direction right X',  'Gaze direction right Y', 'Gaze direction right Z',
 'Gaze point left X (DACSmm)', 'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)',
 'Gaze point right Y (DACSmm)',]

# Convert the type of the column from object to float
for i in Merged[string_replace]:
    Merged[i] = Merged[i].astype(str).str.replace(',', '.').astype(float)
    
    #dividing by 10 to change DCASSmm to mm
    Merged[i] = Merged[i]/10


# In[20]:


gc.collect()


# # Removing unnexessary categorical features

# In[21]:


# Identify categorical features
categorical_features = Merged.select_dtypes(include=['object', 'category']).columns

# Dropping remaining categorical features which are not required
Merged = Merged.drop(categorical_features, axis=1)


# In[22]:


Merged.head()


# In[23]:


Merged.shape


# In[24]:


#Only dropping this columns null value cause cannot fill with mean and its a datetime so i can plot eye tracker recoreded time
Merged.dropna(subset = 'Eyetracker timestamp', inplace = True)


# In[25]:


#Dropping unecessary columns
Merged.drop({'Computer timestamp','Recording duration'},axis = 1,inplace = True)


# In[26]:


Merged.shape


# In[27]:


gc.collect()


# # Handling Null values by fill method

# In[28]:


#Filling all null value with mean
Merged.fillna(Merged.mean(),inplace = True)


# In[29]:


Merged.isnull().sum()


# In[30]:


Merged.dtypes


# # Handling Duplicate values

# In[31]:


#Checking Duplicates
Merged.duplicated().sum()


# In[32]:


#Dropping duplicates
Merged.drop_duplicates(inplace = True)


# In[33]:


gc.collect()


# # Datetime

# In[34]:


def time(x):
    return datetime.datetime.fromtimestamp(x)

#Getting datetime from UNIX timestamp
Merged['Recording_datetime'] = Merged['Recording timestamp'].apply(time)
Merged['Eyetracker_datetime'] = Merged['Eyetracker timestamp'].apply(time)

#Dropping the columns of UNIX timestamp after getting datetime
Merged.drop({'Recording timestamp','Eyetracker timestamp'},axis = 1,inplace = True)


# In[35]:


Merged.head()


# In[36]:


#Timeseries plot on Recording date and time

import matplotlib.pyplot as plt

plt.plot(Merged.Recording_datetime, Merged['Gaze point X'])
plt.plot(Merged.Recording_datetime, Merged['Gaze point Y'])

# set the plot title and axis labels
plt.title('Time Series Plot')
plt.xlabel('Recording Date and Time')
plt.ylabel('Gaze Point X and Y')

# display the plot
plt.show()


# In[37]:


#Timeseries plot on Eyetracking date and time

plt.plot(Merged.Recording_datetime, Merged['Eye movement type'])

plt.plot(Merged.Eyetracker_datetime, Merged['Gaze point X'])
plt.plot(Merged.Eyetracker_datetime, Merged['Gaze point Y'])

# set the plot title and axis labels
plt.title('Time Series Plot')
plt.xlabel('Eye Tracking Date and Time')
plt.ylabel('Gaze Point X and Y')

# display the plot
plt.show()


# # Getting Empathy Score

# In[38]:


Merged.head()


# In[39]:


score1.head(2)


# In[40]:


#Merging with score dataset to get empathy score
Merged = Merged.merge(score1[['Participant nr','Total Score extended','Total Score original']],on = 'Participant nr',how ='inner')


# In[41]:


Merged.head()


# # Correlation

# In[42]:


Merged.corr()


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns

# set figure size
plt.figure(figsize=(10, 10))

sns.heatmap(Merged.corr())

plt.title('Correlation Heatmap')
plt.show()


# In[44]:


Merged.describe()


# In[45]:


#Check skewed data
Merged.hist()
plt.figure(figsize=(30, 30))
plt.show()


# # Empathy Score Prediction

# # Test-Train-Split

# In[46]:


from sklearn.model_selection import train_test_split

#seperating dependent and independent variable x,y
X_Score = Merged.drop({'Total Score original','Recording_datetime','Eyetracker_datetime','Total Score extended','Participant nr'},axis = 1)
y_Score = Merged['Total Score original']

#splitting the data into training 70 % and testing 30%
X_train_score_test, X_test_score_test, y_train_score_test, y_test_score_test = train_test_split(X_Score, y_Score, test_size=0.30, random_state=1)


# # Linear Regression (LR)

# In[47]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_Score, y_Score)
reg.score(X_Score, y_Score)


# In[48]:


reg_predict = reg.predict(X_test_score_test)

print(mean_squared_error(y_test_score_test,reg_predict))


# # Random Forest (RF)

# In[49]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_Score, y_Score)
clf.score(X_Score, y_Score)


# In[50]:


rf_predict = clf.predict(X_test_score_test)

print(mean_squared_error(y_test_score_test,rf_predict))


# In[ ]:





# # Eye Movement Type Classification

# In[51]:


Merged['Eye movement type'].unique()


# In[52]:


Merged['Eye movement type'].value_counts()


# In[53]:


sns.countplot(Merged['Eye movement type'])


# # resampling

# In[54]:


from sklearn.utils import resample

zero = Merged[Merged['Eye movement type'] == 0]
ones = Merged[Merged['Eye movement type'] == 1]
two = Merged[Merged['Eye movement type'] == 2]
three = Merged[Merged['Eye movement type'] == 3]

ones_downsampled = resample(ones, replace=False, n_samples= zero.shape[0] ,random_state=42)
two_downsampled = resample(two, replace=False, n_samples= zero.shape[0] ,random_state=42)
three_downsampled = resample(three, replace=False, n_samples= zero.shape[0] ,random_state=42)


# In[55]:


Final_Merged = pd.concat([zero,ones_downsampled,two_downsampled,three_downsampled])


# In[56]:


Final_Merged.head()


# In[57]:


Final_Merged['Eye movement type'].unique()


# In[58]:


Final_Merged['Eye movement type'].value_counts()


# In[59]:


sns.countplot(Final_Merged['Eye movement type'])


# In[60]:


Final_Merged.head()


# In[61]:


Final_Merged.shape


# In[62]:


gc.collect()


# # Test Train SPlit

# In[63]:


#seperating dependent and independent variable x,y
X = Final_Merged.drop({'Eye movement type','Recording_datetime','Eyetracker_datetime'},axis = 1)
y = Final_Merged['Eye movement type']


# In[64]:


#splitting the data into training 70 % and testing 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# # Model Building

# # Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train,y_train)


# In[66]:


LR_pred = LR.predict(X_test)


# In[67]:


from sklearn.metrics import classification_report

print(classification_report(y_test, LR_pred))


# # Naive bayes

# In[68]:


from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(X_train,y_train)


# In[69]:


NB_pred = NB.predict(X_test)


# In[70]:


print(classification_report(y_test, NB_pred))


# In[ ]:




