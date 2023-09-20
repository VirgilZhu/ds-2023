import math
import random
def monte_carlo(num):
    a = 0
    count = 0
    while a < num:
        x = random.uniform(2, 3)
        y = random.uniform(0, 20)
        if y <= (x*x + 4*x*math.sin(x)):
            count += 1
        a += 1
    return 20*count / num

print(f'定积分结果为{monte_carlo(1000000):.6f}')