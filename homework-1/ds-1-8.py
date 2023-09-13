L = [2, 1, 3, 5, 3]
for i in range(1, 5):
    for j in range(i - 1, -1, -1):
        if L[j]<L[j+1] :
            temp = L[j]
            L[j] = L[j+1]
            L[j+1] = temp
        else:
            break
for i in range(0, 5):
    print(L[i], end = " ")
print("")

L = [2, 5, 4, 5, 3]
start = 1
while start <= len(L) - 1:
    stamp = start - 1
    while stamp >= 0:
        if L[stamp]<L[stamp+1] :
            temp = L[stamp]
            L[stamp] = L[stamp+1]
            L[stamp+1] = temp
            stamp-=1
        else:
            break
    start+=1
for i in range(0, 5):
    print(L[i], end = " ")