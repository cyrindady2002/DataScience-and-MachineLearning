#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


model= pickle.load(open('logm_pkl','rb'))


# In[3]:


st.title('Model Deployment using Logistic Regression')


# In[4]:


import streamlit as st
import pandas as pd

st.title("Manual Data Entry for Titanic Prediction")

# Manual input fields for each feature
Pclass = st.selectbox('Pclass (Ticket Class)', [1, 2, 3], help="Passenger class: 1st, 2nd, or 3rd class")
Age = st.number_input('Age', min_value=0, max_value=100, step=1, help="Age of the passenger")
SibSp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, step=1, help="Number of siblings or spouses aboard")
Parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, step=1, help="Number of parents or children aboard")

# For gender, encode Female and Male as binary features
gender = st.selectbox('Gender', ['Female', 'Male'], help="Gender of the passenger")
Female = 1 if gender == 'Female' else 0
Male = 1 if gender == 'Male' else 0

# For embarkation, one-hot encode the ports
Embarked = st.selectbox('Port of Embarkation', ['C', 'S', 'Q'], help="Port of Embarkation: C = Cherbourg, S = Southampton, Q = Queenstown")
Embarked_C = 1 if Embarked == 'C' else 0
Embarked_S = 1 if Embarked == 'S' else 0
Embarked_Q = 1 if Embarked == 'Q' else 0

# Combine the manual inputs into a DataFrame
input_data = {
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Female': [Female],
    'Male': [Male],
    'Embarked_C': [Embarked_C],
    'Embarked_S': [Embarked_S],
    'Embarked_Q': [Embarked_Q]
}

df_manual = pd.DataFrame(input_data)
st.subheader('Manual Input Data')
st.write(df_manual)

# Assuming 'model' is already defined and trained
if st.button('Predict'):
    pred_prob = model.predict_proba(df_manual)
    pred = model.predict(df_manual)

    # Display the predicted class
    st.subheader('Predicted')
    st.write('Yes' if pred_prob[0][1] > 0.5 else 'No')

    # Display the prediction probabilities
    st.subheader('Prediction Probabilities')
    st.write(pred_prob)


# In[ ]:




