S = input()
s = S[0]
for i in range(1, len(S)):
    if S[i] == s[0]:
        s += S[i]
        if i == len(S) - 1:
            print(s, end = " ") 
        continue
    elif len(s) >= 2:
        print(s, end = " ")
    s = S[i]