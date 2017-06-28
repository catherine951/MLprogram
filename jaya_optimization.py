import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


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


def fitness(data, target, pop, n_cv=5):
    """获取种群个体适应度（分类准确率）。
    :param data: 数据特征值
    :param target: 数据目标值
    :param pop: 参数c,g的实数值矩阵
    :type pop: np.array
    :param n_cv: 交叉验证次数，k-fold
    :type n_cv: int
    :return: 个体适应度列矩阵,每个个体所对应的适应度
    :rtype: np.array
    """
    n_p = pop.shape[0]
    fit = np.zeros([n_p, 1])
    for i in range(n_p):
        clf = SVC(C=pop[i, 0], gamma=pop[i, 1])
        scores = cross_val_score(clf, data, target, cv=n_cv)
        fit[i, 0] = scores.mean() * 100  # 百分比
    return fit


def identify_bw_solution(pop, fit):
    """
    找到种群里面最好的候选人和最差的候选人
    :param pop: 种群
    :param fit: 适应度
    :return: 最优和最差的候选人，最佳适应度
    """
    best_fit = np.max(fit)
    # worst_fit = np.min(fit)
    best_param = pop[np.argmax(fit), :]
    worst_param = pop[np.argmin(fit), :]
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
    :param fit: 原种群适应度
    :param comp_fit: 更改后种群适应度
    :return: new_pop: 新的种群
    """
    n, d = pop.shape
    new_pop = np.zeros([n, d])
    for i in range(n):
        if fit[i, :] >= comp_fit[i, :]:
            new_pop[i, :] = pop[i, :]
        else:
            new_pop[i, :] = comp_pop[i, :]
    return new_pop


def jaya_optimize_svm_cg(n, d, bound, data, target, gm):
    """
    主程序
    :param n: 候选人个数
    :param d: 参数个数
    :param bound: 参数范围
    :param data: 特征向量
    :param target: 目标值向量
    :param gm: 最大迭代次数
    :return: pop, best_param, best_result: 最终种群，最佳参数，最佳适应度
    """
    pop = initialize_population(n, d, bound)
    for g in range(gm):
        fit = fitness(data, target, pop)
        best, worst, best_fit = identify_bw_solution(pop, fit)
        comp_pop = modify_solution(pop, best, worst, bound)
        comp_fit = fitness(data, target, comp_pop)
        new_pop = compare_choose(pop, comp_pop, fit, comp_fit)
        pop = new_pop
        if g >= gm / 2 and best_fit > 95:
            break
    # fit_result = fitness(data, target, pop)
    best_result = best_fit
    # best_result = np.max(fit_result)
    best_param = pop[np.argmax(best_result), :]
    return pop, best_param, best_result

start = time.clock()
data, target = load_iris(return_X_y=True)
bound = np.array([[0, 100], [0, 100]])
n = 10
d = 2
gm = 100
pop, best_p, best_f = jaya_optimize_svm_cg(n, d, bound, data, target, gm)
end = time.clock()
print(pop)
print(best_p)
print(best_f)
print('%d seconds' % (end-start))
