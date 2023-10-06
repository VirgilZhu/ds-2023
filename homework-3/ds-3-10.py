if __name__ == "__main__":
    a = int(input("请输入数："))
    if a == 1 :
        print("1不是质数")
        exit()
    if a == 2 :
        print("2是质数")
        exit()
    if a % 2 == 0:
        print(f'{a}不是质数')
        exit()
    flag = 1
    for i in range(3, int(pow(a, 0.5)) + 1, 2):
        if a % i == 0:
            print(f'{a}不是质数')
            flag = 0
            break
    if flag :
        print(f'{a}是质数')