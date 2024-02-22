import pandas as pd
import random
import csv
import json

# 随机生成病例
def generateCases(filename, num):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(filename, header=None)
    # attributes为存放所有属性的列表
    attributes = df.iloc[0].tolist()

    # 读取除第一行外剩余的数据
    df = pd.read_excel(filename, header=0)
    terms = [df[col].tolist() for col in df.columns]  # 二维数组terms，按列存储
    # lists_dict存放属性名和对应值的列表，供随机生成病例使用
    lists_dict = {attributes[i]: terms[i] for i in range(len(attributes))}

    # 随机生成症状打分文件
    ####################################################################
    diseases = lists_dict["疾病"]
    outerDict = {}
    for i in range(len(diseases)):
        symptom = {}
        for j in range(10):  # 每个疾病找10个症状
            random_row = random.randrange(0, 15)
            random_column = random.randrange(0, 8)
            word = terms[random_row][random_column]
            symptom.update({word: random.randint(1, 100)})
        outerDict.update({diseases[i]: {"index": i, "symptom": symptom}})

    dict_str = str(outerDict)
    # Save the string representation to a .txt file
    file_path = "testScoreFile.txt"
    with open(file_path, 'w') as file:
        file.write(dict_str)
    ####################################################################

    cases = []  # 二维列表
    for i in range(num):
        case = [random.choice(lists_dict[key]) for key in lists_dict]
        cases.append(case)

    with open('cases.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(attributes)  # 如果要去掉表头就把此行注释掉
        for case in cases:
            writer.writerow(case)


if __name__ == '__main__':
    num = 500
    generateCases("0222矩阵.xlsx", num)
    print("生成了", num, "条病例！")
