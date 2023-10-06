def DtoB(dec):
    length = len(dec)
    dec = float(dec)
    print("0.", end="")
    for i in range(20):
        dec *= 2
        print(str(dec)[0], end="")
        dec = float("0" + str(dec)[1:length])

if __name__ == '__main__':
    x = input("请输入十进制小数：")
    DtoB(x)