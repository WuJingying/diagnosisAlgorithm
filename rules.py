import random

import numpy as np

import utils


class main_process:
    def __init__(self, row_size=utils.row_size, column_size=utils.column_size):
        self.last_step = None  # 记录上一步落子位置，坐标形式
        self.row_size = row_size
        self.column_size = column_size
        self.board = np.zeros([self.row_size + 8, self.column_size + 8])  # 前后扩展8个状态？
        self.current_player = 1  # 当前该黑方or白方落子
        self.totalSteps = 0  # 记录总落子数
        self.endFlag = False

    # 重置，棋盘清零，黑方先落子
    def renew(self):
        self.board = np.zeros([self.row_size + 8, self.column_size + 8])
        self.current_player = 1
        self.totalSteps = 0
        self.endFlag = False

    def which_player(self):
        return self.current_player

    # 把棋盘四周向外扩展了4，相当于给原棋盘外面加了一圈
    def current_board_state(self, raw=False):
        if raw:
            return self.board
        else:
            return self.board[4:self.row_size + 4, 4:self.column_size + 4]

    # 模拟重置
    def simulate_reset(self, board_state):  # 这里还可以添加一个检测输入棋盘是否为一局胜负以分的board的算法，但是由于不影响强化学习，所以暂且搁置
        if type(board_state) != np.ndarray:  # 而且还有一部分原因也是因为没有找到一个合适的算法。
            raise ValueError("board_state must be a np array")

        if board_state.shape[0] != self.row_size + 8 or board_state.shape[1] != self.column_size + 8:
            raise ValueError("board size is different from the given size")
        self.board = np.array(board_state, copy=True)

        # step_count: 总计步数，可以对此进行限制
        # black_count: 黑方落子数
        # white_count: 白方落子数
        # line48-59: 计算一个棋盘状态的落子情况
        step_count, black_count, white_count = 0, 0, 0
        for hang in board_state:  # 遍历棋盘
            for lie in hang:  # 行的每一列，1表示黑子，-1表示白子
                if lie == 1:  # 黑方落子
                    step_count += 1
                    black_count += 1
                elif lie == -1:  # 白方落子
                    step_count += 1
                    white_count += 1
                elif lie != 0:
                    raise ValueError(
                        "We got some value wrong in the input board_state, the value in board must be 0, 1, -1")

        if black_count == white_count:  # 黑白落子数相等，说明该黑方落子了
            self.current_player = 1
            self.totalSteps = step_count
        elif black_count == white_count + 1:  # 黑子比白子多1，说明该白方落子了
            self.current_player = -1
            self.totalSteps = step_count
        else:
            raise ValueError("The input board state is not a standard five stone game board, the number of black stone"
                             " and white stone is incorrect")

    # 一次落子
    # place：表示将要落子的位置坐标
    def step(self, tree, place):
        # 这里还需要增加下board state的厚度，因为原版的输入到神经网络里的board state并不只有一个2D matrix。
        # 所以我们需要增加下厚度。
        board = self.board[4:self.row_size + 4, 4:self.column_size + 4]  # 获取棋盘
        if not self.board[place[0] + 4, place[1] + 4] and utils.can_place_in_this_column(self.current_player, board,
                                                                                         place[1]):
            self.board[place[0] + 4, place[1] + 4] = self.current_player  # 落子
            self.last_step = [place[0] + 4, place[1] + 4]  # 记录这次落子的位置
            self.totalSteps += 1  # 总步数+1
            self.current_player = -self.current_player  # 交换玩家

            # 都返回对局状态和当前棋盘
            # 必须落在疾病一列才能分胜负,其他情况都继续对局,除非步数达到上限
            # 若一方落在疾病一列则立刻结束
            if self.check_win(tree):  # 胜负已分
                self.endFlag = True  # 游戏结束
            elif self.totalSteps >= utils.max_step:  # 胜负未分但步数达到上限
                self.endFlag = True  # 游戏结束
            else:  # 胜负未分但步数未达上限
                self.endFlag = False
            return self.board[4:self.row_size + 4, 4:self.column_size + 4]
        else:  # 该位置不可落子, 总步数不增加,重新落子,游戏继续
            self.endFlag = False
            return self.board[4:self.row_size + 4, 4:self.column_size + 4]

    # 判断是否胜负已分
    # 必须在疾病一列有落子,并且下够了5个不同列
    def check_win(self, tree):
        category = ""
        if len(tree.matrixHead) > self.last_step[1]:
            category = tree.matrixHead[self.last_step[1]]  # 获取上一步落子所在列的属性名称
        if utils.isDisease(category):
            # 如果当前落子在疾病一列,则强制对方也在疾病一列落子,然后分别计算双方的分数,结束对局
            player = self.board[self.last_step[0], self.last_step[1]]  # 获取上一步落子的颜色
            # 当前位置可以落子
            if self.last_step[0] < len(tree.matrix) and self.last_step[1] < len(tree.matrix[self.last_step[0]]):
                currentDisease = tree.matrix[self.last_step[0]][self.last_step[1]]  # 获取当前疾病
                diseaseIndex = self.last_step[0] - 4
                score = self.calculate_score(tree, player, currentDisease)  # 当前player的得分
                if score > 0:  # 如果当前方分数大于0，则为有效路径，需要再计算另一方的分数
                    # 对手的疾病落子从剩下的疾病列表中任选一个
                    anotherPlayer = -player  # 对手
                    diseaseList = self.removeStrFromArray(tree.disease, currentDisease)
                    anotherDisease = random.sample(diseaseList, 1)[0]
                    anotherScore = self.calculate_score(tree, anotherPlayer, anotherDisease)
                    if score > anotherScore:  # 需不需要加个经验值，分数大于某个经验值算作？？？
                        return True  # 表示player获胜
        # 没有在疾病一列落子,则胜负未分
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
    # rowNum表示所在的行？
    def calculate_score(self, tree, player, currentDisease):
        board = self.board[4:self.row_size + 4, 4:self.column_size + 4]
        catList = []
        score = 0
        symptomDict = {}
        # 如果这个疾病存在于症状打分表中
        if currentDisease in tree.disease_symptom:
            disease_symptomDict = tree.disease_symptom.get(currentDisease)
            symptomDict = disease_symptomDict.get("symptom")  # 获取这个疾病对应的症状:分数字典
        #  遍历棋盘
        for i in range(len(tree.matrix)):
            for j in range(len(tree.matrix[i])):
                if board[i][j] == player:   # 找到这个player的所有落子
                    matrixVal = tree.matrix[i][j]  # 获取对应位置的术语
                    if matrixVal in symptomDict:  # 如果这个术语在对应疾病的症状:分数字典中
                        score += symptomDict.get(matrixVal)  # 累计这个症状对应的分数

        # return score
                    cat = tree.matrixHead[j]
                    if cat not in catList:
                        catList.append(cat)  # 把这个属性名称加到列表中
        if len(catList) >= utils.max_step and not utils.isContainsDisease(catList):  # 步数超过上限
            return 0
        elif len(catList) >= utils.cat_size and not utils.isContainsDisease(catList):  # 至少历史步骤中已经有5种类型，并且这5种类型中不包含疾病
            score += len(catList)  # 把路径长度累加到奖励得分中
            return score
        else:  # 落子太少
            return 0
