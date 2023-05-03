
# In[1]:

#----------Importing Useful Libraries----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


# Importing the Dataset
data = pd.read_csv('heart.csv')
data.head() 


# In[3]:


#Checking the missing values
data.isnull().any()

# The answer shows that there is no missing values so one of the reason to select this data


# In[4]:


# Get information about the dataframe, including column names and data types
print(data.info())


# In[5]:


# Explratory Data Analysis

# 1. Checking the count of how many people are having or not having heart disease 
sns.countplot(x='target', data = data)
plt.show()

# In[6]:


print('Having disease = ',(data['target'] == 0).sum())
print('Not Having disease = ',(data['target'] == 1).sum())

# The plot and count also shows us one more thing that the data is prety much balanced which is one another reason to select this data

# In[7]:


# Box plots
plt.figure(figsize=(10,7))
sns.boxplot(data=data)
plt.title('Box plots of features in heart.csv')
plt.show()


# In[8]:


# Plot the distribution of age column
sns.distplot(data['age'], kde=False)
plt.title('Distribution of age column')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[9]:


# Scatter plot of age vs maximum heart rate achieved
sns.scatterplot(x='age', y='thalach', data=data, hue='target')
plt.title('Age vs Maximum Heart Rate Achieved')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate Achieved')
plt.show()



# In[10]:


# Doing some Feature Engineering

# First plotting correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix,annot=True)
plt.show()


# In[11]:


# Checking the feautures having high correlation 
high_corr=[] 
mat= correlation_matrix.iloc[:-1]["target"]
for index,i in enumerate(mat):
    if i >= 0.4 or i <= -0.4:
        high_corr.append(mat.index[index])
print('Feaures having High Correlation are: \n',high_corr)


# In[12]:


# Pairplot to visualize the relationships between features having high correlation
sns.pairplot(data, vars=['cp' ,'thalach' ,'exang','oldpeak'], hue='target')
plt.show()


# In[13]:


# Now we are going to implement the Machine Learning Model

# Splittng the Dataset into target and features
X = data.drop("target", axis = 1) 
Y = data["target"]


# In[14]:


# Splittng the Dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)


# In[15]:


# Train the svm classifier 
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)


# In[16]:


# Predict the target values for the test data
y_pred = clf.predict(X_test)


# In[17]:


from sklearn.metrics import accuracy_score

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[18]:


# Plot the SVM output
sns.scatterplot(x=X_test["age"], y=X_test["thal"], hue=y_pred, palette=["red", "blue"])
plt.show()


# In[19]:


# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
