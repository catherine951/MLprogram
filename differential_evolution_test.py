import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import de_optimize_svm_cg


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
    return xx, yy


def test_ds(name):
    xx, yy = load_ds_data(name)
    c, g, f, t = de_optimize_svm_cg(xx, yy)
    plot_result(c, g, f, t)


def test_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    xx, yy = load_breast_cancer(True)
    c, g, f, t = de_optimize_svm_cg(xx, yy)
    plot_result(c, g, f, t)


def plot_result(c, g, fit, trace):
    plt.grid(True)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('fitness curve')
    plt.text(.75*trace.shape[0], .75*fit,
             'best-a: %.3f\nbest-c: %.3f\nbest-g: %.3f' % (fit, c, g))
    plt.plot(trace[:, 0], label='best')
    plt.plot(trace[:, 1], label='average')
    plt.legend(shadow=True, fancybox=True)
    plt.show()


if __name__ == '__main__':
    test_breast_cancer()
    # test_ds('spect')
    # test_ds('glass.data')
    # test_ds('pima')
    # plot_result(np.float64(80), np.float64(3.4444444444444444), np.float64(5.6), np.random.randint(10, 100, size=(100, 2)))
