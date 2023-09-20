#将2001分成尽可能多的3，除非最后只剩下1，如10=3+3+3+1，取10=3+3+2+2

lst = []

def div(n):
    while(n>3):
        if n==4:
            lst.append(2)
            lst.append(2)
            return
        else:
            lst.append(3)
            n-=3
    if n==2:
        lst.append(2)
    return

n = int(input())

div(n)

lst.sort()
print(lst)