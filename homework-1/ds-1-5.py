lst = []
for i in range(0, 4):
    lst.append(int(input()))
lst.sort(reverse = True)
for i in range(0, 4):
    print(lst[i], end = " ")