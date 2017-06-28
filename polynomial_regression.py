# import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from txt_to_useful import txt2poly
from sklearn.pipeline import make_pipeline

X_train, y_train = txt2poly('F:/dataset/mmc1_train.txt')
X_test, y_test = txt2poly('F:/dataset/mmc1_test.txt')

model = make_pipeline(PolynomialFeatures(2), Ridge())
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(y_predict)
"""
colors = ['teal', 'yellowgreen', 'gold']
for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)
"""
plt.plot(y_test, y_test, color='gold', linewidth=2, label="ground truth")
plt.scatter(y_predict, y_test, color='navy', s=30, marker='o', label="testing points")
plt.legend(loc='lower right')
plt.show()
