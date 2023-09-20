import math
import random
def cutting_circle(n):     #割圆法
    side_length = 1     #原边长，也即圆半径
    edge = 6          #原边数
    def length(x):      #新边长
        h = math.sqrt(1-(x / 2)**2)   #h为圆心到side_length的垂直距离
        return math.sqrt((x / 2)**2 + (1-h)**2)
    for i in range(n):
        side_length = length(side_length)
        edge *= 2      #每次边数分割成两倍
        pi = side_length*edge/2  #周长公式
    return edge, pi
edge, pi = cutting_circle(10)
print(f'方法一.割圆法：分割10次，边数为{edge}条，圆周率为{pi:.10f}')

def monte_carlo_pi(num):    #蒙特卡洛法
    a = 0
    count = 0
    while a < num:
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            count += 1
        a += 1
    return 4*count / a   #圆形面积:方形面积
print(f'方法二.蒙特卡洛法：落点1000000次，圆周率为{monte_carlo_pi(1000000):.10f}')

def inf_series_pi(error):   #无穷级数法
    a = 1
    b = 1
    sum = 0
    while 1 / b > error:
        if a % 2 != 0:
            sum += 1 / b
        else:
            sum -= 1 / b
        a += 1
        b += 2
    pi = sum*4   # 1/1-1/3+1/5-1/7+…=Π/4
    return pi, a
pi, times = inf_series_pi(0.0000005)
print(f'方法三.无穷级数法：迭代{times}次，圆周率为{pi:.10f}')
