if __name__ == "__main__":
    score = int(input("请输入成绩："))
    if score < 60 :
        print("不合格")
    elif score < 75 :
        print("合格")
    elif score < 90 :
        print("良好")
    else :
        print("优秀")