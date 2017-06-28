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


def calculate_mean_solution(pop):
    """
    找到种群里面平均值
    :param pop: 种群
    :return: 找到各个参数的平均值
    """
    n, d = pop.shape
    mean = []
    for i in range(d):
        x = pop[:, i]
        mean.append(np.mean(x))
    return mean


def identify_teacher(pop, fit):
    """
    找到种群里的最优值也就是教师
    :param pop: 种群
    :param fit: 适应度
    :return: teacher向量
    """
    best_fit = np.max(fit)
    # worst_fit = np.min(fit)
    teacher = pop[np.argmax(fit), :]
    return teacher, best_fit


def modify_solution(pop, teacher, bound, mean):
    """
    更新种群
    :param pop: 种群
    :param teacher: 教师样本
    :param bound: 参数范围
    :param mean: 参数均值
    :return: 更新后待对比种群
    """
    n, d = pop.shape
    comp_pop = np.zeros([n, d])
    tf = 1 + np.random.random()*(2-1)
    for i in range(n):
        for j in range(d):
            r = np.random.random()
            comp_pop[i, j] = pop[i, j] + r * (teacher[j] - tf * mean[j])
            if comp_pop[i, j] < bound[j, 0] or comp_pop[i, j] > bound[j, 1]:
                comp_pop[i, j] = bound[j, 0] + (bound[j, 1] - bound[j, 0]) * np.random.random()
    return comp_pop


def compare_get_better(pop, comp_pop, fit, comp_fit):
    """
    对比当前种群和更改后种群的适应度，选取适应度较好的形成新的种群
    :param pop: 种群
    :param comp_pop: 更新后待比较的种群
    :param fit: 原种群适应度
    :param comp_fit: 更新后种群适应度
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


def rand_modify_two(pop, fit):
    """
    随机选择两个样本，对比之后更新种群
    :param pop: 种群
    :param fit: 种群适应度
    :return: 新的种群
    """
    n, d = pop.shape
    t = np.random.randint(0, n, size=2)
    if t[0] != t[1]:
        t[1] = np.random.randint(0, n)
    new_pop = np.zeros([n, d])
    for i in range(n):
        for j in range(d):
            r = np.random.random()
            if fit[t[0]] >= fit[t[1]]:
                new_pop[i, j] = pop[i, j] + r * (pop[t[0], j] - pop[t[0], j])
            else:
                new_pop[i, j] = pop[i, j] + r * (pop[t[0], j] - pop[t[0], j])
    return new_pop


def teaching_learning_optimize_svm_cg(n, d, bound, data, target, gm):
    """
    主程序
    :param n: 学生数
    :param d: 参数个数（课程数）
    :param bound: 参数范围
    :param data: 特征向量
    :param target: 目标向量
    :param gm: 最大迭代次数
    :return: final_pop, teacher, best: 最终种群，教师向量，最佳适应度
    """
    pop = initialize_population(n, d, bound)
    for g in range(gm):
        mean = calculate_mean_solution(pop)
        fit = fitness(data, target, pop)
        teacher, best = identify_teacher(pop, fit)
        comp_pop = modify_solution(pop, teacher, bound, mean)
        comp_fit = fitness(data, target, comp_pop)
        new_pop = compare_get_better(pop, comp_pop, fit, comp_fit)
        new_fit = fitness(data, target, new_pop)
        new2_pop = rand_modify_two(new_pop, new_fit)
        new2_fit = fitness(data, target, new2_pop)
        final_pop = compare_get_better(new_pop, new2_pop, new_fit, new2_fit)
        pop = final_pop
        if g >= gm / 2 and best > 95:
            break
    return final_pop, teacher, best


start = time.clock()
data, target = load_iris(return_X_y=True)
bound = np.array([[0, 100], [0, 100]])
n = 10
d = 2
gm = 100
pop_final, best_p, best_f = teaching_learning_optimize_svm_cg(n, d, bound, data, target, gm)
end = time.clock()
print(pop_final)
print(best_p)
print(best_f)
print('%d seconds' % (end-start))
