if __name__ == "__main__":
    A = []
    map = map(int, input("请输入数组A：").split())
    for item in map:
        A.append(item)
    B = A[:]
    n = len(A)
    i = 0
    temp = 1
    while i < n :
        B[i] = temp
        temp *= A[i]
        i += 1
        if i == n :
            break
    i = n - 1
    temp = 1
    while i >= 0 :
        B[i] *= temp
        temp *= A[i]
        i -= 1
        if i == -1 :
            break
    print(B)

