import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def gen_bin_population(n_ind, l_ind):
    """生成初始二进制种群。
    :param n_ind: 个体数量
    :type n_ind: int
    :param l_ind: 个体染色体长度
    :type l_ind: int
    :return: 种群矩阵
    :rtype: np.array
    """
    return np.random.randint(2, size=(n_ind, l_ind))


def get_c_g(pop, v_len, lb, ub):
    """将二进制c,g值转换为对应区间实数值。
    :param pop: 种群（染色体序列组）
    :type pop: np.array
    :param v_len: 参数c和g的染色体长度
    :param lb: 参数c, g上区间
    :param ub: 参数c, g下区间
    :return: 返回c,g的实数值
    :rtype: np.array
    """
    c_g = np.zeros((len(pop), len(v_len)))

    lf = [0]
    lf.extend(v_len)
    lf = np.cumsum(lf)
    for i in range(len(v_len)):
        idx = np.arange(lf[i], lf[i + 1])
        v = np.dot(pop[:, idx], np.power(0.5, np.arange(1, v_len[i] + 1)))
        c_g[:, i] = lb[i] + (ub[i] - lb[i]) * v

    return c_g


def fitness(data, target, n_ind, c_g, n_cv=5):
    """获取种群个体适应度（分类准确率）。
    :param data: 数据特征值
    :param target: 数据目标值
    :param n_ind: 个体数量
    :type n_ind: int
    :param c_g: 参数c,g的实数值矩阵
    :param n_cv: 交叉验证次数，k-fold
    :type n_cv: int
    :return: 个体适应度列矩阵
    :rtype: np.array
    """
    fit = np.zeros((n_ind, 1))
    for i in range(n_ind):
        clf = SVC(C=c_g[i, 0], gamma=c_g[i, 1])
        scores = cross_val_score(clf, data, target, cv=n_cv)
        fit[i, 0] = scores.mean() * 100   # 百分比

    return fit


def ranking(fit):
    """根据适应度值，获取相应的入选率。
    :param fit: 种群个体适应度列矩阵
    :type fit: np.array
    :return: 个体适应度对应入选率列矩阵
    :rtype: np.array
    """
    n_ind = fit.shape[0]
    rat_fun = 2 * np.arange(n_ind) / (n_ind - 1)
    rat = np.empty(fit.shape)
    for i, v in enumerate(np.argsort(fit, axis=0).T[0]):
        rat[v, 0] = rat_fun[i]
    return rat


def select(pop, fit_v, g_gap=0.95):
    """选择合适个体进入下一代（排序法）。
    :param pop: 种群
    :type pop: np.array
    :param fit_v: 种群个体适应度
    :param g_gap: 代沟（入选几率，默认为0.95）
    :type g_gap: float
    :return: 选出的新的种群
    :rtype: np.array
    """
    n_ind, l_ind = pop.shape
    n_sel = int(np.round(g_gap * n_ind + .5))

    new_pop = gen_bin_population(n_sel, l_ind)

    # 排序法选择
    idx = np.argsort(fit_v, axis=0)
    for i in range(n_ind-n_sel, n_ind):
        new_pop[i-n_ind+n_sel, :] = pop[idx[i][0], :].copy()

    # cumfit = np.cumsum(fit_v)  # 适应度累加和
    # for i in range(n_ind):
    #     r = np.random.random() * cumfit[-1]  # 生成一个随机数，在[0,总适应度]之间
    #     first, last = 0, n_ind - 1
    #     mid = int(np.round(n_ind / 2))
    #     idx = -1
    #     # 排中法选择个体
    #     while first <= last and idx == -1:
    #         if r > cumfit[mid]:
    #             first = mid
    #         elif r < cumfit[mid]:
    #             last = mid
    #         else:
    #             idx = mid
    #         mid = int(np.round((first + last) / 2))
    #         if last - first == 1:
    #             idx = last
    #
    #     # 产生新一代个体
    #     new_pop[i, :] = pop[idx, :].copy()

    return new_pop


def crossover(pop, cx_rat=0.7):
    """交叉（两点交叉法）。
    :param pop: 种群
    :type pop: np.array
    :param cx_rat: 交叉概率（默认为0.7）
    :return: 交叉后的种群
    :rtype: np.array
    """
    n_ind, l_ind = pop.shape

    for i in range(0, n_ind, 2):
        if np.random.random() < cx_rat:
            cx_pos = int(np.round(np.random.random() * l_ind))
            if cx_pos < 2:
                continue
            # 对cx_pos及之后的二进制串进行交换
            tmp = pop[i, cx_pos:].copy()
            pop[i, cx_pos:] = pop[i + 1, cx_pos:].copy()
            pop[i + 1, cx_pos:] = tmp

    return pop


def mutate(pop, mut_rat=0.5):
    """变异。
    :param pop: 种群
    :type pop: np.array
    :param mut_rat: 变异概率（默认为0.5）
    :return: 变异后的种群
    :rtype: np.array
    """
    n_ind, l_ind = pop.shape
    for i in range(n_ind):
        if np.random.random() < mut_rat:
            mut_pos = int(np.round(np.random.random() * l_ind-0.5))
            pop[i, mut_pos] = 1 - pop[i, mut_pos]

    return pop


def reins(pop, sel_pop, fit_v, sel_fit_v):
    """将子代个体插入到父代种群中，代替不合适的父代个体.
    :param pop: 父代种群
    :type pop: np.array
    :param sel_pop: 子代种群
    :type sel_pop: np.array
    :param fit_v: 父代种群适应度矩阵
    :type fit_v: np.array
    :param sel_fit_v: 子代种群适应度矩阵
    :type sel_fit_v: np.array
    :return: 重插值后的种群及其适应度
    :rtype: tuple
    """
    assert pop.shape[0] == fit_v.shape[0]
    assert sel_pop.shape[0] == sel_fit_v.shape[0]

    idx = fit_v.argsort(axis=0).T[0]
    idx_sel = sel_fit_v.argsort(axis=0).T[0]
    pop = pop[idx]
    fit_v = fit_v[idx]
    sel_pop = sel_pop[idx_sel][::-1]    # 逆序
    sel_fit_v = sel_fit_v[idx_sel][::-1]

    for i in range(min(pop.shape[0], sel_pop.shape[0])):
        if sel_fit_v[i] >= fit_v[i]:
            pop[i] = sel_pop[i]
            fit_v[i] = sel_fit_v[i]

    return pop, fit_v


def ga_svm_cg_for_classify(data, target, max_gen=200, pop_size=20, g_gap=0.9,
                           c_bound=(0, 100), g_bound=(0, 1000), n_cv=5):
    """用遗传算法优化支持向量分类中的cost和gamma参数。
    :param data: 数据特征值
    :param target: 数据目标值
    :param max_gen: 最大进化代数，一般取值为[100, 500]
    :type max_gen: int
    :param pop_size: 种群个体最大数量，一般取值为[20，100]
    :type pop_size: int
    :param g_gap: 代沟，表示每一代从种群中选择多少个体到下一代
    :type g_gap: float
    :param c_bound: [c_min, c_max], 参数c的变化范围, 默认为(0, 100]
    :type c_bound: tuple, list
    :param g_bound: [g_min, g_max], 参数g的变化范围, 默认为(0, 1000]
    :type g_bound: tuple, list
    :param n_cv: SVM Cross Validation参数, 默认为5
    :type n_cv: int
    :return: 最佳准确率，c，g，参数记录矩阵
    :rtype: tuple
    """
    chromosome_len = 2 * 20

    trace = np.zeros((max_gen, 2))

    pop = gen_bin_population(pop_size, chromosome_len)
    c_g = get_c_g(pop, (20, 20), (c_bound[0], g_bound[0]), (c_bound[1], g_bound[1]))

    obj_v = fitness(data, target, pop_size, c_g, n_cv)

    best_cv_accuracy = np.max(obj_v)
    best_c, best_g = c_g[np.argmax(obj_v), :]

    for gen in range(max_gen):
        fit_v = ranking(obj_v)
        selected = select(pop, fit_v, g_gap)
        selected = crossover(selected)
        selected = mutate(selected)

        c_g = get_c_g(pop, (20, 20), (c_bound[0], g_bound[0]), (c_bound[1], g_bound[1]))
        obj_sel_v = fitness(data, target, len(selected), c_g, n_cv)

        pop, obj_v = reins(pop, selected, obj_v, obj_sel_v)

        new_best_cv_accuracy = np.max(obj_v)
        d_cv = new_best_cv_accuracy - best_cv_accuracy
        cg_tmp = get_c_g(pop, (20, 20), (c_bound[0], g_bound[0]), (c_bound[1], g_bound[1]))
        # print(cg_tmp, obj_v)
        if new_best_cv_accuracy > best_cv_accuracy:
            best_cv_accuracy = new_best_cv_accuracy
            best_c, best_g = cg_tmp[np.argmax(obj_v), :]

        trace[gen, 0] = new_best_cv_accuracy
        trace[gen, 1] = np.average(obj_v)
        # print(best_c, best_g, new_best_cv_accuracy)

        if gen >= max_gen / 2 and best_cv_accuracy >= 95 and d_cv <= 0.01:
            break

    return best_cv_accuracy, best_c, best_g, trace

if __name__ == '__main__':
    # test_b_c()
    # test_spec()
    pass
