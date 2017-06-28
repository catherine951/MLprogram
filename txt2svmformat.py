xc = []
f = open('F:/dataset/mmc1.txt', 'r')
for line in f.readlines():
    line = line.split('\t')
    xc.append(line)
print(xc)
print(len(xc[0]))
path = 'F:/dataset/'
name = input("你要创建的文件的名字：\n")
full_path = path + name + '.txt'
fw = open(full_path, 'w')
for i in range(len(xc)):
    for j in range(len(xc[i])):
        if j == 0:
            fw.write(xc[i][j])
        else:
            fw.write(str(j))
            fw.write(':')
            fw.write(xc[i][j])
        if j+1 < len(xc[i]):
            fw.write(' ')
    fw.write('\n')

