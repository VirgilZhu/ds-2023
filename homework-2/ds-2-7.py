def Cubic_root_3():
    c = 10
    g = c/2
    i = 0
    while (abs(g**3 - c) > 0.00000000001):
        g = (2 * g + c / (g * g)) / 3
        i += 1
        print("%d:%.13f" % (i, g))

Cubic_root_3()