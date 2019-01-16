
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from sklearn.preprocessing import Imputer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import t
from scipy import stats
# import CustStat as stat
from math import pow, sqrt


def clean():

    bc = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target, random_state = 0)
    scaler = MinMaxScaler(feature_range=(-10.0, 10.))
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    svm = SVC()
    svm.fit(x_train_scaled, y_train)

    print('The accuracy on training subset: {:.3f}'.format(svm.score(x_train_scaled, y_train)))
    print('The accuracy on test subset: {:.3f}'.format(svm.score(x_test_scaled, y_test)))

    probs = svm.predict(x_test_scaled)
    print(probs)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for breast cancer classifier')
    plt.grid(True)
    plt.show()


def main():
    clean()


if __name__ == "__main__":
    main()
