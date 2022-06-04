import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn.utils import shuffle


data = pd.read_csv('student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

print(data.head())

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""training 30 models and saving model with highest accuracy to pickle"""
"""
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print(accuracy)
    
    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""


"""Using pickle to load in previously trained model for ease of development"""
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

"""Plotting results in 2d space"""
x_axis = np.linspace(0, 100, len(x))
#plt.scatter(x_test[:, 0], y_test, color='green')
#print(linear.predict(x_train))
#plt.plot(x_test[:, 0], linear.predict(x_test), color='orange')
#plt.show() 

"""printing out prediction results """
predictions = linear.predict(x_test)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use("ggplot")
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()