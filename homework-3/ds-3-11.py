import time
import random

if __name__ == "__main__":
    start = time.time()
    A = [random.random() for i in range(0, 100000)]
    end = time.time()
    execution_time = end - start
    print(f'随机生成100000个0-1之间的小数，耗时：{execution_time}')
