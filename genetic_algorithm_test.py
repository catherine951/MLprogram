import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import ga_svm_cg_for_classify


def test_ds(name='spect'):
    import csv
    xx = []
    yy = []
    with open('datasets\\' + name) as f:
        data = csv.reader(f)
        for count, row in enumerate(data):
            xx.append([float(i) for i in row[1:]])
            yy.append(int(row[0]))
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    a, c, g, t = ga_svm_cg_for_classify(xx, yy, c_bound=(0, 1))
    plot_result(a, c, g, t)

    # from sklearn.svm import SVC
    # from sklearn.model_selection import cross_val_score
    # f = open(name + '-rst.tsv', 'w')
    # for i in np.arange(0.1, 100, 0.1):
    #     for j in np.arange(0.1, 1000, 0.1):
    #         f.write('{0}\t{1}\t{2}\n'.format(i, j, cross_val_score(SVC(C=i, gamma=j), xx, yy, cv=5).mean()))
    # f.close()


def test_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    xx, yy = load_breast_cancer(True)
    a, c, g, t = ga_svm_cg_for_classify(xx, yy)
    plot_result(a, c, g, t)


def plot_result(a, c, g, trace):
    plt.grid(True)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('fitness curve')
    plt.text(.75*trace.shape[0], .75*a,
             'best-a: %.3f\nbest-c: %.3f\nbest-g: %.3f' % (a, c, g))
    plt.plot(trace[:, 0], label='best')
    plt.plot(trace[:, 1], label='average')
    plt.legend(shadow=True, fancybox=True)
    plt.show()


if __name__ == '__main__':
    # test_breast_cancer()
    # test_ds('spect')
    test_ds('monks')
    # test_ds('pima')
    # plot_result(np.float64(80), np.float64(3.4444444444444444), np.float64(5.6), np.random.randint(10, 100, size=(100, 2)))
