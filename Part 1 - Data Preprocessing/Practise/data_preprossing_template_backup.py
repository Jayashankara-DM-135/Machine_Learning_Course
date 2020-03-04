#Data Preproccssing
#Importing libarary 

import numpy as np # for math related stuffs
import matplotlib.pyplot as plt # Ploting graphs ext
import pandas as pd #handling data sets

#import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
print(X)
Y = dataset.iloc[:, 3].values
print(Y)

#Taking care of missing data
print("Before handling the missing data")
print(X)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

#Categorical data
"""
Means attributes which devides the date set into different sections
In Given dataset Country and Purchased attribute are categorically divide
the data.

In python index starts from zero, Hence 0 is used to represent country 
couloum. 
"""

from sklearn import preprocessing
labelencoder_X = preprocessing.LabelEncoder()
#We are considering only country colum now
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print(X)

"""
Now country coulum is converted into nummeric , like 0, 1 and 2
But this number just a identity, nothing related to one is greter 
then other.
To avoid the confusion arise by these number.
We are cretaing separate colum for each country by craeting a dummy 
cloums
"""
onehotencoder = preprocessing.OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print(X)


"""
Now consider purchase categorical colum and conver into numberic,
But no need to transfrom them into array, since Machine learing know 
It is dependent varible.
"""
labelencoder_Y = preprocessing.LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(Y)

#Splitting the dataset into the Traning set and Test set.
"""
X is independe varible and Y is depended varible.
So we chosed some part form X for testing and remaing for traning
Simillarly for independed varible.
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_tarin, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#feature Scaling
"""
Means two attributes are not on same scale.
Here age and salary are on different scale, In machine learing it's
better to take same scale for both.

For example if we take Ecluerian disatake for two obervation 
The result is always is dominated by salary since it is scaling is higher than
age.

France	48.0	79000.0 ==> consider as a point1 (X1, Y1)
Germany	30.0	54000.0 ==> consider as a point2 (X2, Y2)

ED = squrerootof(pow((x2-x1),2) - pow((y2-y1),2))
Results alwaysnegative since salary is huge.

----------------
There are two mnethod for feature scaling:
    a> Satndardisation
    b> Normalisation
------------------

Note: Basically convert age values range from 0 to 1 and
same way for salary. 

When you call StandardScaler.fit(X_train), what it does is calculate the mean and 
variance from the values in X_train. 
Then calling .transform() will transform all of the features by subtracting the mean 
and dividing by the variance. For convenience, these two function calls can be done
 in one step using fit_transform().

The reason you want to fit the scaler using only the training data is because you 
don't want to bias your model with information from the test data.

If you fit() to your test data, you'd compute a new mean and variance for 
each feature. In theory these values may be very similar if your test and train 
sets have the same distribution, but in practice this is typically not the case.

Instead, you want to only transform the test data by using the parameters 
computed on the training data.

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#Below order matters since fit is calcualted mean and variance henec 
# Both are used in X_test as well.
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#No need for independ varible Y , since it is scaling form o to 1.
# since depended varible here is classification , I mean o or 1
# but if it's regression like varies from one number to some large number
# in that case we need to apply the scaling on depend varible as well.

"""
IMP: Machine learning module is not always based on Ecludin distance.
For Example decision tree is not based on Ecludin disatnce, but still need
scalling.

"""





