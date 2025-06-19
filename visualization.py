import imgvision as iv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

HSI=scipy.io.loadmat('/home/hyperspectralSR/CEGATSR/mcodes/dataset/Cave_x2/evals/block_balloons_ms_1.mat')
HSI2=(HSI['gt'])
# HSI = np.load('/home/zhangmj/hyperspectralSR/CEGATSR/datasets/Chikusei_x2/test/Chikusei_test.mat',allow_pickle=True)
print(HSI2.shape)



convertor = iv.spectra()

RGB = convertor.space(HSI2)

plt.imshow(RGB)
plt.show()