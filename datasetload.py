import numpy as np


def load_ds_csv(name):
    import csv
    xx = []
    yy = []
    with open('F:/dataset/' + name) as f:
        data = csv.reader(f)
        for count, row in enumerate(data):  # enumerate列举、枚举
            xx.append([float(i) for i in row[1:]])
            yy.append(float(row[0]))
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    return xx, yy


def load_ds_xlsx(name):
    """有表头的Excel文件，最后一列是目标值，前几列是特征值
    :param name:
    :type name:
    :return xx,yy:
    :type xx,yy:
    """
    import pandas as pd
    xx = []
    yy = []
    with pd.ExcelFile('F:/dataset/' + name) as xls:
        data = pd.read_excel(xls, 'Sheet1')
        df = data.values  # numpy.ndarray
        for count, row in enumerate(df):  # enumerate列举、枚举
            xx.append([float(i) for i in row[:-1]])
            yy.append(float(row[-1]))
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    return xx, yy


def load_ds_data(name):
    """
    data文件
    :param name:
    :return:
    """
    xxx = []
    xx = []
    yy = []
    with open('F:/dataset/UCI/' + name) as f:
        data = f.read()
        data = data.split('\n')
        for count, row in enumerate(data):
            xxx.append([i for i in row.split(',')])
        for i in range(len(xxx)):
            xn = []
            for j in range(len(xxx[i])):
                if(j<len(xxx[i])-1):
                    xn.append(float(xxx[i][j]))
                else:
                    yy.append(xxx[i][j])
            xx.append(xn)
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    return xx, yy


def load_ds_mat(name, x_index, y_index):
    import scipy.io as scio
    f = 'F:/dataset/UCI/' + name
    data = scio.loadmat(f)
    print(data.keys())
    x_index = input("特征名：")
    y_index = input("目标名：")
    return data[x_index], data[y_index]


# x, y = load_ds_csv('125.csv')
# x, y = load_ds_xlsx('1251.xlsx')
# x, y = load_ds_data('iris.data')
# x, y = load_ds_mat('fisheriris.mat')
# print(x)
# print(y)



