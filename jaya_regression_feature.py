import time
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from datasetload_for_mac import load_ds_xlsx


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
data, target = load_ds_xlsx('125.xlsx')
bound = np.array([[0, 100], [0, 100]])
n = 10
d = 2
gm = 100
X_test = np.array([[1.316833333, 0.065875883, 0.12893797, 14.89668823, 0.45875305, 8.333333333, 3.432707757],
          [1.264285714, 0.061258363, 0.115846485, 16.17829698, 0.475585603, 7.285714286, 4.311238279],
          [1.276035088, 0.0493137, 0.11875003, 15.4452509, 0.31606147, 8.122807018, 6.351258423],
          [1.282338983, 0.054909985, 0.126242176, 15.76180627, 0.322937569, 7.983050847, 5.227611937],
          [1.288229508, 0.059215207, 0.132308387, 15.92421372, 0.329002738, 7.852459016, 4.541418169]])

y_test = np.array([451, 580, 265, 325, 455])

pop, best_p, best_f, best_r = jaya_optimize_svr_cg(n, d, bound, data, target, gm, X_test, y_test)

end = time.clock()
print(pop)
print(best_p)
print(best_f)
print(best_r)
print('%d seconds' % (end-start))

