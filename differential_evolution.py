import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


def gen_bin_population(n_p, d, bound):
    """生成初始二进制种群。
    :param n_p: 个体数量
    :type n_p: int
    :param d: 问题规模，个体染色体长度,是c和g的染色体长度之和
    :type d: int
    :param bound: c和g的上限和下限，bound = np.array([[0, 100], [0, 100]])
    :type bound: np.array
    :return: 种群矩阵
    :rtype: np.array
    """
    pop = np.zeros([n_p, d])
    for i in range(d):
        for j in range(n_p):
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


# 变异操作有不同的操作方法，这种是比较常见的一种策略
def mutate(pop, f0, bound):
    """变异过程。
    :param pop: 种群矩阵
    :type pop: np.array
    :param f0: 缩放因子（默认为0.5）
    :type f0: float
    :param bound: c和g的上限和下限，bound = np.array([[0, 100], [0, 100]])
    :type bound: np.array
    :return: 中间个体v
    :rtype: np.array
    """
    n_p, d = pop.shape
    # f = f0 * 2 ** (np.e ** (1 - gm / (gm + 1 - g)))  # 使用了自适应算子
    f = f0
    v = np.zeros([n_p, d])
    for i in range(n_p):
        # 随机选取三个个体,并保证与i不相同
        r = np.random.randint(0, n_p, size=3)
        if r[0] == i:
            r[0] = np.random.randint(0, n_p)
        elif r[1] == i:
            r[1] = np.random.randint(0, n_p)
        elif r[2] == i:
            r[2] = np.random.randint(0, n_p)
        v[i, :] = pop[r[0], :] + f * (pop[r[1], :] - pop[r[2], :])
        for j in range(d):
            if v[i, j] < bound[j, 0] or v[i, j] > bound[j, 1]:  # 防止溢出
                v[i, j] = bound[j, 0] + (bound[j, 1] - bound[j, 0]) * np.random.random()  # 若溢出则重新选择
    return v


def crossover(pop, v, cr):
    """交叉过程。
    :param pop: 种群矩阵
    :type pop: np.array
    :param v: 中间个体
    :type v: np.array
    :param cr: 交叉概率
    :type cr: float
    :return: 交叉后的个体u
    :rtype: np.array
    """
    n_p, d = pop.shape
    u = np.zeros([n_p, d])
    for i in range(n_p):
        reserveone = np.random.randint(0, d)
        for j in range(d):
            if np.random.random() <= cr or j == reserveone:
                u[i, j] = v[i, j]
            else:
                u[i, j] = pop[i, j]
    return u


def select(pop, u, data, target):
    """选择过程。
    :param pop: 种群矩阵
    :type pop: np.array
    :param u: 交叉后的个体
    :type u: np.array
    :param data: 数据特征值
    :param target: 数据目标值
    :return: 新一代的种群new_pop
    :rtype: np.array
    """
    n_p, d = pop.shape
    new_pop = np.zeros([n_p, d])
    u_fit = fitness(data, target, u)
    x_fit = fitness(data, target, pop)
    for i in range(n_p):
        if u_fit[i] <= x_fit[i]:
            new_pop[i, :] = u[i, :]
        else:
            new_pop[i, :] = pop[i, :]
    return new_pop


def de_optimize_svm_cg(data, target, gm=10, n_p=20, d=2,
                           bound=np.array([[0, 100], [0, 100]]), cr=0.5, n_cv=5):
    """主程序。
    :param data: 数据特征值
    :param target: 数据目标值
    :param gm: 最大进化代数
    :type gm: int
    :param n_p: 种群规模
    :type n_p: int
    :param d: 问题规模
    :type d: int
    :param bound: c和g的上限和下限，bound = np.array([[0, 100], [0, 100]])
    :type bound: np.array
    :param cr: 交叉概率
    :type cr: float
    :param n_cv: 交叉验证次数
    :type n_cv: int
    :return: best_c, best_g, fit_best, trace
    :rtype: np.array
    """
    pop = gen_bin_population(n_p, d, bound)
    trace = np.zeros([gm, 2])
    fit = fitness(data, target, pop, n_cv=5)
    fit_best = np.max(fit)
    best_c, best_g = pop[np.argmax(fit_best), :]
    g = 1
    for g in range(gm):
        v = mutate(pop, f0, bound)
        u = crossover(pop, v, cr)
        new_pop = select(pop, u, data, target)
        new_fit = fitness(data, target, new_pop, n_cv=5)
        new_fit_best = np.max(new_fit)
        if new_fit_best > fit_best:
            fit_best = new_fit_best
        fit_best_index = np.argmax(fit_best)
        best_c, best_g = new_pop[fit_best_index, :]
        trace[g, 0] = fit_best
        trace[g, 1] = np.average(new_fit)
        if g >= gm / 2 and fit_best >= 95:
            break
    return best_c, best_g, fit_best, trace


# 涉及到的算法参数
gm = 10000
f0 = 0.5
n_p = 100
cr = 0.9
d = 2
data, target = load_iris(return_X_y=True)
bound = np.array([[0, 100], [0, 100]])
pop = gen_bin_population(10, 2, bound)
v = mutate(pop, 0.5, bound)
u = crossover(pop, v, cr)
new_pop = select(pop, u, data, target)
c, g, f, t = de_optimize_svm_cg(data, target)
print(c)
print(g)
print(f)
print(t)
