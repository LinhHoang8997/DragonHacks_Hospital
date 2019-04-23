
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB


# In[71]:


struct_df = pd.read_csv("Structural_Measures_-_Hospital.csv",dtype={'Provider ID': str})
struct_df.replace(["Not Available",""," "],np.nan,inplace=True)


# In[72]:


pay_df = pd.read_csv("Payment_and_value_of_care_-_Hospital.csv",dtype={'Provider ID': str,'ZIP Code': str})
pay_df.replace(["Not Available",""," "],np.nan,inplace=True)


# In[73]:


general_df = pd.read_csv("Hospital_General_Information.csv",dtype={'Provider ID': str})
general_df.replace(["Not Available",""," "],np.nan,inplace=True)

general_df.loc[:,"Hospital overall rating"] = general_df.loc[:,"Hospital overall rating"].astype(float)


# In[74]:


pay_df = pay_df.drop(['Location','Phone number'],axis=1);
pay_df.info()


# In[75]:


pay_df.loc[:,"Payment"] = pd.to_numeric(pay_df["Payment"].apply(lambda x: str(x).replace("$","").replace(",","")),errors='coerce')
pay_df.loc[:,"Lower estimate"] = pd.to_numeric(pay_df["Lower estimate"].apply(lambda x: str(x).replace("$","").replace(",","")),errors='coerce')
pay_df.loc[:,"Higher estimate"] = pd.to_numeric(pay_df["Higher estimate"].apply(lambda x: str(x).replace("$","").replace(",","")),errors='coerce')

pay_df['Payment'].head(8)


# In[76]:


pay_df[["Payment","Lower estimate","Higher estimate"]].head(5)


# In[77]:


pay_df_index = pay_df.iloc[:,:7]
pay_df_index.columns


# ## Get the Payment estimates out

# In[78]:


# Payment itself
tf_pay = pay_df.pivot(index="Provider ID", columns='Payment measure name', values='Payment').reset_index()
# Lower Estimatetf_pay_l
tf_pay_l = pay_df.pivot(index="Provider ID", columns='Payment measure name', values='Lower estimate').reset_index()
# Lower Estimate
tf_pay_h = pay_df.pivot(index="Provider ID", columns='Payment measure name', values='Higher estimate').reset_index()


# In[79]:


placehold = tf_pay.columns.tolist()
for index, text  in enumerate(tf_pay.columns[1:]):
    placehold[index+1] = str(text) + " - actual"
tf_pay.columns = placehold
tf_pay.head(1)


# In[80]:


placehold = tf_pay_l.columns.tolist()
for index, text  in enumerate(tf_pay_l.columns[1:]):
    placehold[index+1] = str(text) + " - lower"
tf_pay_l.columns = placehold
tf_pay_l.head(1)


# In[81]:


placehold = tf_pay_h.columns.tolist()
for index, text  in enumerate(tf_pay_h.columns[1:]):
    placehold[index+1] = str(text) + " - higher"
tf_pay_h.columns = placehold
tf_pay_h.head(1)


# ## Get Value of Care Respond

# In[82]:


df_value = pay_df.iloc[:,-6:-1]
df_value = pd.concat([pay_df_index,df_value],axis=1)
df_value.columns


# In[83]:


tf_value = df_value.pivot(index="Provider ID", columns='Value of care display ID', values='Value of care category').reset_index()
tf_value.rename(columns={'MORT_PAYM_30_AMI': 'MORT_PAYM_30_HEART ATT', 'MORT_PAYM_30_PN': 'MORT_PAYM_30_PNEU','MORT_PAYM_30_HF': 'MORT_PAYM_30_HEART FAIL'}, inplace=True)


# In[84]:


tf_value.head(2)


# In[85]:


for col in tf_value[2:]:
    x = tf_value[str(col)].str.split('and', 1, expand=True)


# In[86]:


lst_value_df = []
for item in tf_value.columns.tolist()[1:len(tf_value.columns)]:
#     print(item)
    df = tf_value[item].str.split('and', 1, expand=True)
    df.columns = [item + "_1",item + "_2"]
    lst_value_df.append(df)


# In[87]:


lst_value_df.append(tf_value.iloc[:,:1])
# for i in lst_value_df:
tf_value_split = pd.concat(lst_value_df, axis=1)


# In[88]:


tf_value_split.head(2)


# In[89]:


tf_value_split.iloc[:,2].unique()


# ## Get Structural Measures

# In[90]:


struct_df.info()


# In[91]:


struct_df.head(2)


# In[92]:


# Payment itself
tf_struct = struct_df.pivot(index="Provider ID", columns='Measure Name', values='Measure Response').reset_index()


# In[93]:


tf_struct.head(1)


# In[94]:


tf_struct = tf_struct.replace({"Yes": 1,"No": 0})
tf_struct.head(2)


# ## Combine all dataframes

# In[95]:


tf_index = pay_df_index.drop_duplicates()


# In[96]:


# Start with Index
complete = tf_index

list_of_df = [tf_value_split,tf_struct,tf_pay,tf_pay_l,tf_pay_h,general_df]
for each in list_of_df:
    each.loc[:,'Provider ID'] = each.loc[:,'Provider ID'].apply(str)
for each in list_of_df:
    complete = complete.merge(each,on="Provider ID",how="left")


# In[97]:


complete.head(10)


# In[98]:


print("Fields to apply hot one encoding on:")
for col in complete: 
    if str(complete.loc[:,col].dtype).startswith(("float",'int'))== 0:
        complete.loc[:,col] = complete.loc[:,col].astype(str)
        print(complete.columns.get_loc(col),":", col)


# In[99]:


com_coded = complete.copy()

lst_to_onehot = [28,29,30,31,32,33,34,35,37]

# Create dummy variables of 0 and 1 as a way to on hot code categorical variables without any ordinal purpose
dummies = pd.get_dummies(com_coded[com_coded.columns[lst_to_onehot]],dummy_na=False);


# In[100]:


com_coded = pd.concat([com_coded.drop(com_coded.columns[lst_to_onehot],axis=1),dummies],axis=1)


# In[101]:


com_coded.head(3)
com_coded.shape


# In[102]:


# print("Fields to apply label encoding on:")
# for col in com_coded: 
#     if str(com_coded.loc[:,col].dtype).startswith(("float",'int'))== 0:
#         com_coded.loc[:,col] = com_coded.loc[:,col].astype(str)
#         print(com_coded.columns.get_loc(col),":", col)


# In[103]:


labelencoder = LabelEncoder()
for i in range(1,15,1):
    com_coded.iloc[:, i] = labelencoder.fit_transform(com_coded.iloc[:, i])


# In[104]:


com_coded.set_index("Provider ID", inplace=True)


# In[105]:


# print("Fields to apply label encoding on:")
# for col in com_coded: 
#     if str(com_coded.loc[:,col].dtype).startswith(("float",'int'))== 0:
#         com_coded.loc[:,col] = com_coded.loc[:,col].astype(str)
#         print(com_coded.columns.get_loc(col),":", col)


# In[106]:


com_coded.iloc[:,32:57] = com_coded.iloc[:,32:57].astype(float)
# com_coded = com_coded.astype(float)


# In[107]:


com_coded_fina = com_coded.fillna(com_coded.mean())
# com_coded_fina.info()


# In[108]:


# hospital_overall_rating


# ## Develop Machine Learning Model for predicting categories

# Try a dirty, fast model first:
# - Naive Bayes
# 
# Then Random Tree
# 
# Then XGBoost if needed

# In[109]:


# TEST CORRLINEARILATIY 
corr = com_coded.corr()

# plt.figure(figsize=(10,10))
# cmap = sns.diverging_palette(300, 10, as_cmap=True)
# sns.heatmap(corr,cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .2})
# # Find a correlation table that is less complicated visually
# plt.show()
# # This heatmap is for understanding colinearality 


# Minimize highly correlated variables

# In[110]:


# Select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.90
to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

# Then drop those columns
com_coded_fina = com_coded_fina.drop(to_drop_corr, axis=1)


# In[111]:


print("After filling null values and dropping highly correlated features, we have:")
print(com_coded_fina.shape)
print("A drop from")
print(com_coded.shape, "in the original completed dataframe")


# In[112]:


check = []
for i in com_coded_fina.columns:
    if "PAYM" in i:
        check.append(i)


# In[113]:


features = com_coded_fina.drop(check, axis=1)


# In[114]:


# PUT TARGET HERE
labels = com_coded_fina.loc[:,"MORT_PAYM_30_PNEU_2"]
# features = com_coded_fina.drop("MORT_PAYM_30_HEART FAIL_1", axis=1)


# In[115]:


from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.15, random_state = 42)


# In[116]:


guass_NB = GaussianNB()
guass_NB.fit(train_features,train_labels)


# In[117]:


# COMP_PAYM_90_HIP_KNEE_1                                                                    4784 non-null int32
# COMP_PAYM_90_HIP_KNEE_2                                                                    4784 non-null int32
# MORT_PAYM_30_HEART ATT_1                                                                   4784 non-null int32
# MORT_PAYM_30_HEART ATT_2                                                                   4784 non-null int32
# MORT_PAYM_30_HEART FAIL_1                                                                  4784 non-null int32
# MORT_PAYM_30_HEART FAIL_2                                                                  4784 non-null int32
# MORT_PAYM_30_PNEU_1                                                                        4784 non-null int32
# MORT_PAYM_30_PNEU_2


# In[118]:


predicted_labels = guass_NB.predict(test_features)
predicted_probability = guass_NB.predict_proba(test_features)


# In[119]:


# Use classification_report to compare the test labels (what we know to be true) and the predicted labels generated by
# the Gaussian Naive Bayes
print(metrics.classification_report(test_labels, predicted_labels))


# In[120]:


print(metrics.confusion_matrix(test_labels, predicted_labels))


# In[121]:


from sklearn import metrics


# In[122]:


print("Accuracy:",metrics.accuracy_score(predicted_labels, test_labels))


# ### Model Results

# Model works relatively well with Predicting the following:
# - Complications due to Hip Replacement - COMP_PAYM_90_HIP_KNEE_1
# - Payment to patients for Hip Replacement - COMP_PAYM_90_HIP_KNEE_2
# - Mortality for patients due to Heart failure - MORT_PAYM_30_HEART FAIL_1 
# - Payment to patients for Heart failure - MORT_PAYM_30_HEART FAIL_2 
# - Mortality for patients due to Pneumonia - MORT_PAYM_30_PNEU_1 
# - Payment to patients for Pneumonia - MORT_PAYM_30_PNEU_2
# 
# **With accuracy >75%**

# In[123]:


#Use joblib to save the logreg model for later use'
filename = 'finalized_gaussNB_model.sav'
joblib.dump(guass_NB, filename)


# ## Develop ML Model for predicting scores

# In[124]:


# PUT TARGET HERE
target = "Hospital overall rating"

# Transform the df into ints
com_coded_int = com_coded_fina.astype(int)

labels = com_coded_int.loc[:,target]
features = com_coded_int.drop(target, axis=1)


# In[125]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.15, random_state = 42)


# In[126]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(train_features, train_labels)


# In[127]:


predicted_labels = logreg.predict(test_features)


# In[128]:


# confusion_matrix = confusion_matrix(test_labels, predicted_labels)
# print(confusion_matrix)


# In[129]:


print(classification_report(test_labels, predicted_labels))


# In[130]:


print("Accuracy:",metrics.accuracy_score(predicted_labels, test_labels))


# In[131]:


#Use joblib to save the logreg model for later use'
filename = 'finalized_logreg_model.sav'
joblib.dump(logreg, filename)


# ## Integrate into an app that predicts

# In[132]:


bench_value = pd.DataFrame(com_coded.mean())
# Strangely, I only thought of this as test_features is a dataframe, and it worked....


# In[133]:


predict_one = pd.Series(guass_NB.predict(bench_value), index=com_coded.columns.values.tolist())


# In[134]:


predict_one_prob = pd.DataFrame(guass_NB.predict_proba(bench_value).T,columns=com_coded.columns.values.tolist())


# In[135]:


predict_one_prob


# In[136]:


# COMP_PAYM_90_HIP_KNEE_1                                                                    4784 non-null int32
# COMP_PAYM_90_HIP_KNEE_2                                                                    4784 non-null int32
# MORT_PAYM_30_HEART ATT_1                                                                   4784 non-null int32
# MORT_PAYM_30_HEART ATT_2                                                                   4784 non-null int32
# MORT_PAYM_30_HEART FAIL_1                                                                  4784 non-null int32
# MORT_PAYM_30_HEART FAIL_2                                                                  4784 non-null int32
# MORT_PAYM_30_PNEU_1                                                                        4784 non-null int32
# MORT_PAYM_30_PNEU_2 


# In[137]:


# predict_one_prob

