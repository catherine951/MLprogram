import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# sns.set(style='whitegrid', context='notebook')
# file = pd.ExcelFile('/Users/maomao/Downloads/1251.xlsx')
file = pd.ExcelFile('F:/dataset/1251.xlsx')
# file = pd.ExcelFile('F:/dataset/mmc2.xlsx')
df = pd.read_excel(file, 'Sheet1')
cols = df.columns
cols = list(cols)

# 画散点图，分析各个特征值之间的变化趋势
sns.pairplot(df[cols], size=2.5)
plt.show()

# 画出热点图，分析特征值之间的相关性
cm = np.corrcoef(df[cols].values.T)
print(cm)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()
