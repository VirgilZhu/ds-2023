import re

def match_id_card(id_card):
    pattern = r'^(\d{6})(\d{4})(\d{2})(\d{2})(\d{3})(\d|X)$'
    result = re.match(pattern, id_card)
    if result:
        print("身份证号码有效")
        print("行政区划代码：", result.group(1))
        print("出生日期：", result.group(2), result.group(3), result.group(4))
        print("顺序码：", result.group(5))
        print("校验码：", result.group(6))
    else:
        print("身份证号码无效")

if __name__ == '__main__':
    id_card = input("输入身份证号：")
    match_id_card(id_card)
