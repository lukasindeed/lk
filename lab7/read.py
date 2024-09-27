import scipy.io
import numpy as np

# 读取 .m 文件
data = scipy.io.loadmat('data.m')

# 提取矩阵
A4 = data['A4']
B4 = data['B4']
A8 = data['A8']
B8 = data['B8']

# 保存为文本文件
np.savetxt('A4.txt', A4, fmt='%d')
np.savetxt('B4.txt', B4, fmt='%d')
np.savetxt('A8.txt', A8, fmt='%d')
np.savetxt('B8.txt', B8, fmt='%d')

