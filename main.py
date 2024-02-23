import os
import time
import warnings

import matplotlib.pyplot as plt

import torch
import json
import utils
from network import neuralnetwork as nn
from new_MCTS import MCTS

warnings.filterwarnings("ignore")

matrixFile = '0222矩阵.xlsx'
diseaseScoreFile = 'symptomScores.txt'


# train函数完成模型训练
def train(tree_file=None, pretrained_model=None, game_file_saved_dict="game_record_2"):
    # 保存游戏记录的路径
    if not os.path.exists(game_file_saved_dict):
        os.mkdir(game_file_saved_dict)

    # pretrained_model = ""   # 若有模型可接着训练
    # 加载预训练模型
    if pretrained_model:
        Net = torch.load(pretrained_model)
    # 创建神经网络
    else:
        Net = nn(input_layers=3, row_size=utils.row_size, column_size=utils.column_size,
                 learning_rate=utils.learning_rate)

    stack = utils.random_stack()
    # 默认tree_file为None，执行else后的语句，创建蒙特卡洛树
    if tree_file:
        tree = utils.read_file(tree_file)
    else:
        stack = utils.random_stack()
        # 默认tree_file为None，执行else后的语句，创建蒙特卡洛树
        if tree_file:
            tree = utils.read_file(tree_file)
        else:
            tree = MCTS(row_size=utils.row_size, column_size=utils.column_size, neural_network=Net,
                        matrixFile=matrixFile,
                        diseaseScoreFile=diseaseScoreFile)

    Net.adjust_lr(1e-3)  # 设置神经网络学习率
    record = []
    game_time = 1  # game_time从1开始

    # 开始训练
    print("game begins...")
    while True:
        # game_record包含每一步的distribution和action
        # 每个game_record代表一局游戏，长度为奇数表示最后一个落子的是黑方
        game_record, eval, steps = tree.game()   # 完成一次对局
        if len(game_record) % 2 == 1:  # game_record长度为奇数，黑方获胜
            print("game {} completed, black win, this game length is {}".format(game_time, len(game_record)))
        else:  # game_record长度为偶数，白方获胜
            print("game {} completed, white win, this game length is {}".format(game_time, len(game_record)))
        print("The average eval:{}, the average steps:{}".format(eval, steps))
        # 将game_record写入文件
        utils.write_file(game_record,
                         game_file_saved_dict + "/" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime()) + '.pkl')
        # 根据game_record，产生训练数据。参数为对局记录和棋盘大小
        train_data = utils.generate_training_data(game_record=game_record, row_size=utils.row_size,
                                                  column_size=utils.column_size)

        # 将训练数据放入stack中
        for i in range(len(train_data)):
            stack.push(train_data[i])
        # 加载训练数据到my_loader
        my_loader = utils.generate_data_loader(stack)
        utils.write_file(my_loader, "debug_loader.pkl")  # 将myloader写入debug_loader中

        if game_time % 50 == 0:  # 每模拟50次对局，完成一次神经网络训练
            for _ in range(5):
                record.extend(Net.train(my_loader, game_time))  # 训练
            print("train finished")  # 一次训练完成

        # 保存一次模型
        if game_time % 100 == 0:  # 每完成2次训练，保存一次模型。并保存入test_game_record中
            torch.save(Net, "model_{}.pkl".format(game_time))
            test_game_record, _, _ = tree.game(train=False)
            utils.write_file(test_game_record, game_file_saved_dict + "/" + 'test_{}.pkl'.format(game_time))
            print("We finished a test game at {} game time".format(game_time))

        # 每100次对局,输出一次loss record图像
        # if game_time % 100 == 0:
        #     plt.figure()
        #     plt.plot(record)
        #     plt.title("cross entropy loss")
        #     plt.xlabel("step totalSteps")
        #     plt.ylabel("Loss")
        #     plt.savefig("loss record_{}.svg".format(game_time))
        #     plt.close()

        game_time += 1  # 更新对局次数


# 使用对局记录进行模型训练
def train_use_record_file():
    my_loader = utils.read_file("debug_loader.pkl")
    game_time = 75
    Net = nn(input_layers=3, row_size=utils.row_size, column_size=utils.column_size, learning_rate=utils.learning_rate)
    tree = MCTS(row_size=utils.row_size, column_size=utils.column_size, neural_network=Net)
    matrixHead, disease, matrix = utils.readMatrix(matrixFile)
    tree.matrixHead = matrixHead
    tree.matrix = matrix
    tree.disease = disease
    # 加载疾病打分文件
    tree.disease_symptom = {}
    with open(diseaseScoreFile, encoding='utf-8') as f:
        fs = ''.join(f)
        disease_symptom = json.loads(fs)
    tree.disease_symptom = disease_symptom

    Net.adjust_lr(1e-3)  # 设置神经网络学习率
    record = []
    for _ in range(5):
        record.extend(Net.train(my_loader, game_time))  # 训练
    print("train finished")  # 一次训练完成
    torch.save(Net, "model_{}.pkl".format(game_time))


if __name__ == '__main__':
    train()
    # print("finish")

