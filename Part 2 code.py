# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 01:03:12 2019

@author: Sriram
"""

import pandas as pd
from scipy import stats
# Import input data
dataa=pd.read_csv("HR-Employee-Attrition.csv")
dataa
# Check data balancing
dataa['Attrition'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt 
sns.countplot(x='Attrition',data=dataa,palette='hls')
plt.show()
# Scale data
from sklearn.preprocessing import MinMaxScaler
# Remove categorical variables before scaling
categorical_columns = ['Attrition','BusinessTravel','Department','Education','EducationField',\
                       'EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel',\
                      'JobRole','JobSatisfaction','MaritalStatus','Over18',\
                      'OverTime','PerformanceRating','RelationshipSatisfaction',\
                      'StockOptionLevel','WorkLifeBalance']

data1=dataa.drop(categorical_columns,axis=1)
x = dataa.loc[:, categorical_columns]
x
# Scale numeric variables
scaler = MinMaxScaler(feature_range=(0, 1))
Xzz = pd.DataFrame(scaler.fit_transform(data1),columns = data1.columns)
Xzz
# Combin scaled numeric variables with categorical variables
result = pd.concat([Xzz, x], axis=1, sort=False)
result
# Perform boxcox to reduce skewness in dataset
columns=result.columns
columns
for i in range(len(columns)):
    try:
        if abs(stats.skew(result[columns[i]]))>0.5:
            result[columns[i]]=list(stats.boxcox(result[columns[i]])[0])
    except:
        print('skipped '+str(i)+' column')

for i in result.columns:
    
    try:
        print(stats.skew(result[i]))
        
    except:
        print('categorical variable')
# Set datatype of categorical variables as category      
for column in categorical_columns:
    result[column] = result[column].astype('category')
# Separate prdictor variables  
X = result.iloc[:,result.columns!='Attrition']
# Create dummy variable for categorical data
dummy_data=pd.get_dummies(X)
dummy_data
# View total features
features=dummy_data.columns
features
len(features)

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

# Perform PCA for dimensionality reduction
pca = PCA(n_components=35)
principalComponents = pca.fit_transform(dummy_data)
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()

# Label Yes = 1 and No = 0 for response variable
Yq=result['Attrition']
le = preprocessing.LabelEncoder()
le.fit(Yq)
Yq=le.transform(Yq)

# Split data using PCA components in place of predictor variables
X_trains, X_tests, Y_trains, Y_tests = train_test_split(principalComponents, Yq,random_state=42, test_size=0.30)

# Perform balancing on training dataset
resampling=SMOTE(sampling_strategy='auto')
xx,yy=resampling.fit_sample(X_trains,Y_trains.ravel())
pd.DataFrame(yy)[0].value_counts().plot(kind='bar')

# BernoulliNB modelling
bnb = BernoulliNB(binarize=0.0)
bnb.fit(xx, yy)
yp=bnb.predict(X_tests)
# Check overfitting
train_score = bnb.score(xx, yy)
test_score = bnb.score(X_tests, Y_tests)
# Display results
print("\n--- BERNOULLI NAÏVE BAYES ----\n") 
print('Train Score: {} \nTest Score: {}'.format(train_score, test_score))
print('\nAccuracy score:\n', accuracy_score(Y_tests,yp))
print('\nConfusion Matrix:\n', confusion_matrix(Y_tests,yp))
print('\nClassification Report:\n', classification_report(Y_tests,yp))





