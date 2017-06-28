"""nature那篇文章的数据用Jaya优化的SVR预测"""
print(__doc__)


import time
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from datasetload import load_ds_xlsx
from sklearn.feature_selection import VarianceThreshold


def initialize_population(n, d, bound):
    """
    :param n: 种群大小
    :param d: 参数个数
    :param bound: 参数范围
    :type bound: np.array
    :return: pop: 初始种群
    """
    pop = np.zeros([n, d])
    for i in range(d):
        for j in range(n):
            pop[j, i] = bound[i, 0] + (bound[i, 1] - bound[i, 0]) * np.random.random()
    return pop


def svr(data, target, X_test, c, g):
    """
    支持向量回归。
    :param data: 训练集特征向量
    :param target: 训练集目标向量
    :param X_test: 测试集特征向量
    :param c: 惩罚因子
    :param g: gamma
    :return: y_predict: 预测结果
    """
    clf = svm.SVR(C=c, gamma=g)  # rbf, ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    # clf = svm.NuSVR()  # Nurbf
    # clf = LinearSVR()  # LinearSVR
    # SVR(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
    # cache_size=200, verbose=False, max_iter=-1)
    clf.fit(data, target)
    y_predict = clf.predict(X_test)
    return y_predict


def fitness(data, target, pop, X_test, y_test):
    """
    得到测试集的误差列向量
    :param data: 训练集特征向量
    :param target: 训练集目标向量
    :param pop: 种群
    :param X_test: 测试集特征向量
    :param y_test: 测试集目标向量
    :return: 误差列向量
    """
    n_p = pop.shape[0]
    n_test, d_test = X_test.shape
    # fit = np.zeros([n_p, 1])
    mse = np.zeros([n_p, 1])  # mean_squared_error
    y_predict = np.zeros([n_p, n_test])
    for i in range(n_p):
        clf = svm.SVR(C=pop[i, 0], gamma=pop[i, 1])
        clf.fit(data, target)
        for j in range(n_test):
            y_predict[i, j] = clf.predict(X_test[j, :])
            mse[i, :] += (y_test[j] - y_predict[i, j])**2
        mse[i, :] = mse[i, :]/n_test
        # scores = cross_val_score(clf, data, target, cv=n_cv)
        # fit[i, 0] = scores.mean() * 100  # 百分比
    return mse


def identify_bw_solution(pop, mse):
    """
    找到种群里面最好的候选人和最差的候选人
    :param pop: 种群
    :param mse: 误差列向量
    :return: 最优和最差的候选人，最小误差
    """
    best_fit = np.min(mse)
    # worst_fit = np.min(fit)
    best_param = pop[np.argmin(mse), :]
    worst_param = pop[np.argmax(mse), :]
    return best_param, worst_param, best_fit


def modify_solution(pop, best_param, worst_param, bound):
    """

    :param pop: 种群
    :param best_param: 最佳候选人
    :param worst_param: 最差候选人
    :param bound: 参数范围
    :return: 更改后待比较的种群
    """
    n, d = pop.shape
    comp_pop = np.zeros([n, d])
    for i in range(d):
        r1 = np.random.random()
        r2 = np.random.random()
        for j in range(n):
            comp_pop[j, i] = pop[j, i] + r1 * (best_param[i] - abs(pop[j, i])) - r2 * (worst_param[i] - abs(pop[j, i]))
            if comp_pop[j, i] < bound[i, 0] or comp_pop[j, i] > bound[i, 1]:
                comp_pop[j, i] = bound[i, 0] + (bound[i, 1] - bound[i, 0]) * np.random.random()
    return comp_pop


def compare_choose(pop, comp_pop, fit, comp_fit):
    """
    对比当前种群和更改后种群的适应度，选取适应度较好的形成新的种群
    :param pop: 种群
    :param comp_pop: 更改后待比较的种群
    :param fit: 原种群误差
    :param comp_fit: 更改后种群误差
    :return: new_pop: 新的种群
    """
    n, d = pop.shape
    new_pop = np.zeros([n, d])
    for i in range(n):
        if fit[i, :] <= comp_fit[i, :]:
            new_pop[i, :] = pop[i, :]
        else:
            new_pop[i, :] = comp_pop[i, :]
    return new_pop


def jaya_optimize_svr_cg(n, d, bound, data, target, gm, X_test, y_test):
    """
    主程序
    :param n: 候选人个数
    :param d: 参数个数
    :param bound: 参数范围
    :param data: 训练集特征向量
    :param target: 训练集目标向量
    :param gm: 最大迭代次数
    :param X_test: 测试集特征向量
    :param y_test: 测试集目标向量
    :return: pop, best_param, best_result, best_result: 最终种群，最佳参数组，最小误差，预测结果（y_predict）
    """
    pop = initialize_population(n, d, bound)
    for g in range(gm):
        fit = fitness(data, target, pop, X_test, y_test)
        best, worst, best_fit = identify_bw_solution(pop, fit)
        comp_pop = modify_solution(pop, best, worst, bound)
        comp_fit = fitness(data, target, comp_pop, X_test, y_test)
        new_pop = compare_choose(pop, comp_pop, fit, comp_fit)
        pop = new_pop
        if g >= gm / 2 and best_fit < 1:
            break
    # fit_result = fitness(data, target, pop)
    # best_result = np.max(fit_result)
    # best_result = best_fit
    best_param = pop[np.argmin(best_fit), :]
    best_result = svr(data, target, X_test, best_param[0], best_param[1])
    return pop, best_param, best_fit, best_result

start = time.clock()
# data, target = load_iris(return_X_y=True)
# data, target = load_ds_xlsx('mmc2.xlsx')
data, target = load_ds_xlsx('mmc3.xlsx')
bound = np.array([[0, 100], [0, 100]])
n = 10
d = 2
gm = 100
"""
X_test = np.array([[4.00000, 0.93843, 162.63200, 135.54500, 1.72401, 6.96800, 0.40788, 24.98600, 53.28727, 0.23461, 40.65800, 33.88625, 0.43100, 1.74200, 0.10197, 6.24650, 13.32182],
                   [4.00000, 0.94004, 162.67400, 135.60800, 1.72716, 6.99600, 0.40902, 25.14000, 53.66100, 0.23501, 40.66850, 3.90200, 0.43179, 1.74900, 0.10225, 6.28500, 13.41525],
                   [4.00000, 0.93925, 162.66300, 135.58300, 1.72573, 6.98200, 0.40857, 25.07200, 53.49799, 0.23481, 40.66575, 33.89575, 0.43143, 1.74550, 0.10214, 6.26800, 13.37450],
                   [4.00000, 0.93770, 162.73500, 135.67000, 1.72120, 6.93000, 0.40854, 24.93000, 53.25440, 0.23443, 40.68375, 33.91750, 0.43030, 1.73250, 0.10214, 6.23250, 13.31360],
                   [3.00000, 0.94750, 167.50000, 138.75000, 1.79750, 7.00000, 0.46750, 29.50000, 65.21250, 0.31583, 55.83333, 46.25000, 0.59917, 2.33333, 0.15583, 9.83333, 21.73750]])

y_test = np.array([29.43, 85.36, 59.01, -46.68, 182.90])
"""


X_test = np.array([[1.72401, 6.96800, 0.40788],
                   [1.72716, 6.99600, 0.40902],
                   [1.72573, 6.98200, 0.40857],
                   [1.72120, 6.93000, 0.40854],
                   [1.79750, 7.00000, 0.46750]])
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = sel.fit_transform(X_test)
X = X.tolist()
y_test = np.array([29.43, 85.36, 59.01, -46.68, 182.90])


clf = svm.SVR()
clf.fit(data, target)
y_predict = clf.predict(X_test)


pop, best_p, best_f, best_r = jaya_optimize_svr_cg(n, d, bound, data, target, gm, X_test, y_test)
end = time.clock()
print(pop)
print(best_p)
print(best_f)
print("Jaya优化后的拟合结果")
print(best_r)
print("直接用SVR的拟合结果")
print(y_predict)
print('%d seconds' % (end-start))

