# file = open('F:/dataset/mmc1.txt', 'r')


def txt2poly(filepath):
    file = open(filepath, 'r')
    xc = []
    x = []
    y = []
    for line in file.readlines():
        line = line.strip('\n')
        line = line.split('\t')
        xc.append(line)
    for i in range(len(xc)):
        xn = []
        for j in range(len(xc[i])):
            if j == 0:
                y.append(float(xc[i][j]))
            else:
                xn.append(float(xc[i][j]))
        x.append(xn)
    return x, y
