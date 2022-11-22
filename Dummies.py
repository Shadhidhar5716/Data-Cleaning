import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pandas.read_csv(r'C:/Users/Shashi/Documents/DataSets/animal_category.csv')

data.dtypes

data.info()


data.describe()

# 1st B.M

data.mean()

data.median()

data.mode()


# 2nd B.M

data.var()

data.std()


# 3rd B.M

data.skew()


# 4th B.M

data.kurt()



# Graphical Representation


plt.boxplot(data)

plt.hist(data)




data.dtypes


duplicates = data.duplicated()
duplicates.sum()


# Dummy variables

#data.drop(['Index'], axis=1, inplace=True)

data_new = pandas.get_dummies(data)


data_new1 = pandas.get_dummies(data, drop_first = True)


# data.columns()


data = data[['Animals', 'Gender', 'Homly', 'Types']]


# One Hot Encoding

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

enc_data = pandas.DataFrame(enc.fit_transform(data.iloc[:,2:]).toarray())


# Label Encoding

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()


# Data splite into input, output labels

X = data.iloc[:,:4]
Y = data.iloc[:,1]


X['Animals'] = labelencoder.fit_transform(X['Animals'])

X['Gender'] = labelencoder.fit_transform(X['Gender'])

X['Homly'] = labelencoder.fit_transform(X['Homly'])

X['Types'] = labelencoder.fit_transform(X['Types'])


















