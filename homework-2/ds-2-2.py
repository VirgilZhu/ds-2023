n = 10
base = 2**10
num = []
num.append(base)
print('2**10 = {}'.format(base))
for i in range(20, 60, 10):
    num.append(base * num[i//10 - 2])
    print('2**{} = {}'.format(i, num[i//10 - 1]))
