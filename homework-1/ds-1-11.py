def deCubic(n):
    left = 0
    right = n
    mid = (left + right)/2
    precision = 0.000001
    while(abs(n - mid**3) >= precision):
        if n < mid**3:
            right = mid
        else:
            left = mid
        mid = (left + right)/2
    print('%.3f' % mid)
deCubic(3)