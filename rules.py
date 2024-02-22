import random

import numpy as np

import utils


class main_process:
    def __init__(self, row_size=utils.row_size, column_size=utils.column_size, AI=None):
        self.row_size = row_size
        self.column_size = column_size
        self.board = np.zeros([self.row_size + 8, self.column_size + 8])  # 前后扩展8个状态？
        self.current_player = 1
        self.passed = 0
        if AI:
            self.AI = "random"

    def renew(self):
        self.board = np.zeros([self.row_size + 8, self.column_size + 8])
        self.current_player = 1
        self.passed = 0

    def vs_AI(self):
        pass

    def which_player(self):
        return self.current_player

    def current_board_state(self, raw=False):
        if raw:
            return self.board
        else:
            return self.board[4:self.row_size + 4, 4:self.column_size + 4]  # 这里是+4

    def simulate_reset(self, board_state):  # 这里还可以添加一个检测输入棋盘是否为一局胜负以分的board的算法，但是由于不影响强化学习，所以暂且搁置
        if type(board_state) != np.ndarray:  # 而且还有一部分原因也是因为没有找到一个合适的算法。
            raise ValueError("board_state must be a np array")
        if board_state.shape[0] != self.row_size + 8 or board_state.shape[1] != self.column_size + 8:
            raise ValueError("board size is different from the given size")
        self.board = np.array(board_state, copy=True)
        step_count, black_count, white_count = 0, 0, 0
        for hang in board_state:
            for lie in hang:
                if lie == 1:
                    step_count += 1
                    black_count += 1
                elif lie == -1:
                    step_count += 1
                    white_count += 1
                elif lie != 0:
                    raise ValueError(
                        "We got some value wrong in the input board_state, the value in board must be 0, 1, -1")
        if black_count == white_count:
            self.current_player = 1
            self.passed = step_count
        elif black_count == white_count + 1:
            self.current_player = -1
            self.passed = step_count
        else:
            raise ValueError("The input board state is not a standard five stone game board, the number of black stone"
                             " and white stone is incorrect")

    def step(self, tree, place):  # 这里还需要增加下board state的厚度，因为原版的输入到神经网络里免得board state可并不是只有一个2D matrix。
        self.passed += 1  # 所以我们需要增加下厚度。
        if not self.board[place[0] + 4, place[1] + 4]:
            self.board[place[0] + 4, place[1] + 4] = self.current_player
            self.current_player = -self.current_player
            self.last_step = [place[0] + 4, place[1] + 4]
            # 不知道+几
            if self.check_win(tree):
                return False, self.board[4:self.row_size + 4, 4:self.column_size + 4]
            elif self.passed == self.row_size * self.column_size:
                return None, self.board[4:self.row_size + 4, 4:self.column_size + 4]
            return True, self.board[4:self.row_size + 4, 4:self.column_size + 4]
        else:
            raise ValueError("here already has a stone, you can't please stone on it")

    # 判断是否获胜
    def check_win(self, tree):
        cat = ""
        if len(tree.matrixHead) > self.last_step[1]:
            cat = tree.matrixHead[self.last_step[1]]
        if utils.isDisease(cat):
            # 当前落子方 1为黑方，2为白方
            currentStep = self.board[self.last_step[0], self.last_step[1]]
            if self.last_step[0] < len(tree.matrix) and self.last_step[1] < len(tree.matrix[self.last_step[0]]):
                currentDisease = tree.matrix[self.last_step[0]][self.last_step[1]]
                score = self.calculate_score(tree, currentStep, currentDisease, self.last_step[0])
                if score > 0:  # 如果当前方分数大于0，则为有效路径，需要再计算另一方的分数
                    anotherStep = -currentStep
                    diseaseList = self.removeStrFromArray(tree.disease, currentDisease)
                    anotherDisease = random.sample(diseaseList, 1)[0]
                    anotherDiseaseIndex = tree.disease.index(anotherDisease)
                    anotherScore = self.calculate_score(tree, anotherStep, anotherDisease, anotherDiseaseIndex)
                    if score > anotherScore:  # 需不需要加个经验值，分数大于某个经验值算作？？？
                        return True
        return False

    # 移除数组中的一个元素
    def removeStrFromArray(self, arr, str):
        arrnew = []
        for val in arr:
            if val != str:
                arrnew.append(val)
        return arrnew


    # 计算当前路径的得分
    # 如果路径上的症状在症状打分表中存在且有得分，则加到score中
    def calculate_score(self, tree, currentStep, currentDisease, rowNum):
        matrix = self.board
        catList = []
        score = 0
        symptomDict = {}
        if currentDisease in tree.disease_symptom:
            disease_symptomDict = tree.disease_symptom.get(currentDisease)
            symptomDict = disease_symptomDict.get("symptom")
        for i in range(len(tree.matrix)):
            for j in range(len(tree.matrix[i])):
                val = matrix[i][j]
                if val == currentStep and i != rowNum and j != self.last_step[1]:
                    matrixVal = tree.matrix[i][j]
                    if matrixVal in symptomDict:
                        score += symptomDict.get(matrixVal)
                    cat = tree.matrixHead[j]
                    if cat not in catList:
                        catList.append(cat)
        if len(catList) >= utils.max_step and not utils.isContainsDisease(catList):  # 步数超过上限
            return 0
        elif len(catList) >= utils.cat_size and not utils.isContainsDisease(catList):  # 至少历史步骤中已经有5种类型，并且这5种类型中不包含疾病
            return score
        else:
            return 0
