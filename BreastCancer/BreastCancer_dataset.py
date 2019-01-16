
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from sklearn.preprocessing import Imputer
from sklearn.datasets import load_breast_cancer
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import t
from scipy import stats
# import CustStat as stat
from math import pow, sqrt


def clean():

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df = df[(df != 0).all(1)]
    writer = ExcelWriter('BreastCancer.xlsx')
    df.to_excel(writer, 'Sheet1')
    writer.save()

def main():
    clean()


if __name__ == "__main__":
    main()
