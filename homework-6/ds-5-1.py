import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    samples = np.random.normal(0, 1, 10000)
    samples = np.sort(samples)
    left = samples[0:9999:2]
    right = samples[9999:0:-2]
    samples = np.concatenate([left, right], axis=0)
    print(samples)
    plt.rcParams['font.sans-serif'] = ['Simsun']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('10000个服从标准正态分布的随机样本散点图')
    ax1.scatter(np.linspace(0, 10000, 10000), samples,  c='red')
    ax1.set_ylabel('样本大小', c='red')
    ax1.set_xticks(np.arange(0, 10001, 1000))
    ax1.grid()
    ax2.set_title('10000个服从标准正态分布的随机样本频率直方图')
    ax2.hist(samples, density=True, bins=100)
    ax2.set_ylabel('频率', c='blue')
    plt.tight_layout()
    ax2.grid()
    plt.show()
