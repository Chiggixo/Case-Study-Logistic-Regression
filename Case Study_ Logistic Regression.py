#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('Titanic-Dataset.csv')


# In[3]:


data.head()


# In[4]:


data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()


# In[5]:


X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[8]:


y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[9]:


y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[ ]:




