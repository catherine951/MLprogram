import numpy as np
import matplotlib.pyplot as plt
from particle_swarm_optimization import pso_svm_cg_for_classify


def test_ds(name='spect'):
    import csv
    xx = []
    yy = []
    with open('dataset\\'+name) as f:
        data = csv.reader(f)
        for count, row in enumerate(data):
            xx.append([float(i) for i in row[1:]])
            yy.append(int(row[0]))
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    fitness_best, p_best, g_best, trace = pso_svm_cg_for_classify(xx, yy, max_gen=50,
                                                                  c_bound=(0.001, 100), g_bound=(0.001, 100))
    print('best: ', fitness_best)
    print(p_best)
    print(g_best)
    plot_result(p_best, g_best, trace)


def test_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    xx, yy = load_breast_cancer(True)
    fitness_best, p_best, g_best, trace = pso_svm_cg_for_classify(xx, yy, max_gen=50,
                                                                  c_bound=(0.001, 100), g_bound=(0.001, 100))
    print('best: ', fitness_best)
    print(p_best)
    print(g_best)
    plot_result(p_best, g_best, trace)


def plot_result(p_best, g_best, trace):
    plt.grid(True)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('fitness curve')
    plt.text(.85, .85, 'best-c: {0}\nbest-g: {1}'.format(g_best[0], g_best[1]))
    plt.plot(trace[0, :], label='c')
    plt.plot(trace[1, :], label='g')
    plt.plot(trace[2, :], label='fitness')
    plt.legend(shadow=True, fancybox=True)
    plt.show()


if __name__ == '__main__':
    test_breast_cancer()
    # test_ds('spect')
    # test_ds('monks')
    # test_ds('pima')
