import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def get_fitness(data, target, c, g, n_cv=5):
    """获取粒子适应度（分类准确率）。
    :param data: 数据特征值
    :param target: 数据目标值
    :param c: 参数c
    :type c: float
    :param g: 参数g
    :type g: float
    :param n_cv: 交叉验证次数，k-fold
    :type n_cv: int
    :return: 粒子适应度列矩阵
    :rtype: np.array
    """
    clf = SVC(C=c, gamma=g)
    scores = cross_val_score(clf, data, target, cv=n_cv)
    return scores.mean() * 100
    # return -(c*np.sin(c)+g*np.cos(g))


def get_fitness_all(data, target, swarm, n_cv=5):
    """获取粒子适应度（分类准确率）。
    :param data: 数据特征值
    :param target: 数据目标值
    :param swarm: 个体数量
    :type swarm: np.array
    :param n_cv: 交叉验证次数，k-fold
    :type n_cv: int
    :return: 粒子适应度列矩阵
    :rtype: np.array
    """
    fit = np.zeros(swarm.shape[1])
    for i in range(swarm.shape[1]):
        fit[i] = get_fitness(data, target, swarm[0, i], swarm[1, i], n_cv)

    return fit


def pso_svm_cg_for_classify(data, target, max_gen=200, swarm_size=20, k=0.1, w_v=1, w_p=1, c1=2, c2=2,
                            c_bound=(0.001, 100), g_bound=(0.001, 1000), n_cv=5):
    """用粒子群算法优化支持向量分类中的cost和gamma参数。
       v = w_v*v + c1*r1*(p_best - p) + c2*r2*(g_best - p)
       p = p + w_p*v
    :param data: 数据特征值
    :param target: 数据目标值
    :param max_gen: 最大迭代次数，一般取值为[100, 500]
    :type max_gen: int
    :param swarm_size: 粒子群最大粒子数，一般取值为[20，100]
    :type swarm_size: int
    :param k: 速率和x的关系(V = kX)，应取 [0.1, 1.0]，默认0.1
    :type k: float
    :param w_v: 速度惯性权重，默认为1，一般取 [0.8,1.2]
    :type w_v: float
    :param w_p: 位置惯性权重，默认为1，一般取 [0.8,1.2]
    :type w_p: float
    :param c1: pso参数局部搜索能力，默认为2
    :type c1: float
    :param c2: pso参数全局搜索能力，默认为2
    :type c2: float
    :param c_bound: [c_min, c_max], 参数c的变化范围, 默认为(0.001, 100]
    :type c_bound: tuple, list
    :param g_bound: [g_min, g_max], 参数g的变化范围, 默认为(0.001, 1000]
    :type g_bound: tuple, list
    :param n_cv: SVM Cross Validation参数, 默认为5
    :type n_cv: int
    :return: (最佳准确率，c，g，参数记录矩阵)
    :rtype: tuple
    """
    trace = np.zeros((3, max_gen))     # 记录c, g的轨迹
    v_c_max = k * (c_bound[1] - c_bound[0])
    v_g_max = k * (g_bound[1] - g_bound[0])

    swarm = np.asarray([np.random.rand(swarm_size) * (c_bound[1] - c_bound[0]) + c_bound[0],
                        np.random.rand(swarm_size) * (g_bound[1] - g_bound[0]) + g_bound[0]],
                       np.float64)     # 生成c, g的粒子群，大小为2*swarm_size
    swarm_v = np.asarray([np.random.rand(swarm_size) * v_c_max,
                          np.random.rand(swarm_size) * v_g_max],
                         np.float64)   # 生成c, g的速度，大小为2*swarm_size
    fitness = get_fitness_all(data, target, swarm, n_cv)  # 初始适应度
    fitness_best = np.max(fitness)

    p_best = swarm.copy()
    g_best = swarm[:, np.argmax(fitness)]

    for gen in range(max_gen):
        print(gen)
        for i in range(swarm_size):
            # 更新速度
            swarm_v[:, i] = swarm_v[:, i] * w_v + \
                            c1*np.random.random()*(p_best[:, i] - swarm[:, i]) + \
                            c2*np.random.random()*(g_best - swarm[:, i])
            if swarm_v[0, i] > v_c_max:
                swarm_v[0, i] = v_c_max
            if swarm_v[0, i] < -v_c_max:
                swarm_v[0, i] = -v_c_max
            if swarm_v[1, i] > v_g_max:
                swarm_v[1, i] = v_g_max
            if swarm_v[1, i] < -v_g_max:
                swarm_v[1, i] = -v_g_max

            # 更新粒子
            swarm[:, i] += swarm_v[:, i] * w_p
            if swarm[0, i] > c_bound[1]:
                swarm[0, i] = c_bound[1]
            if swarm[0, i] < c_bound[0]:
                swarm[0, i] = c_bound[0]
            if swarm[1, i] > g_bound[1]:
                swarm[1, i] = g_bound[1]
            if swarm[1, i] < g_bound[0]:
                swarm[1, i] = g_bound[0]

            # 粒子变异，变异率为0.5
            if np.random.random() > 0.5:
                k = np.random.randint(2)
                if k:
                    swarm[k, i] = (g_bound[1]-g_bound[0])*np.random.random()+g_bound[0]
                else:
                    swarm[k, i] = (c_bound[1]-c_bound[0])*np.random.random()+c_bound[0]

            fitness_i = get_fitness(data, target, swarm[0, i], swarm[1, i], n_cv)

            # 粒子最优更新
            if fitness_i > fitness[i]:
                p_best[:, i] = swarm[:, i]
                fitness[i] = fitness_i

            # 粒子群最优更新
            if fitness_i > fitness_best:
                g_best = swarm[:, i]
                fitness_best = fitness_i

        trace[0, gen] = g_best[0]
        trace[1, gen] = g_best[1]
        trace[2, gen] = np.max(fitness)

    return fitness_best, p_best, g_best, trace
