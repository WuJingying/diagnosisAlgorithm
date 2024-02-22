import json
import sys
import time

import numpy as np

import utils
from rules import main_process as five_stone_game

num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
            13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

distrib_calculater = utils.distribution_calculater(utils.row_size, utils.column_size)


class edge:
    def __init__(self, action, parent_node, priorP):
        self.action = action
        self.counter = 1.0
        self.parent_node = parent_node
        self.priorP = priorP
        self.child_node = None  # self.search_and_get_child_node()

        self.action_value = 0.0

    def backup(self, v):  # back propagation
        self.action_value += v
        self.counter += 1
        self.parent_node.backup(-v)

    def get_child(self):
        if self.child_node is None:
            self.counter += 1
            self.child_node = node(self, -self.parent_node.node_player)
            return self.child_node, True
        else:
            self.counter += 1
            return self.child_node, False

    def UCB_value(self):  # 计算当前的UCB value
        Q = self.action_value / self.counter
        # if self.action_value:  # 有非0值
        #     Q = self.action_value / self.counter
        # else:
        #     Q = 0
        return Q + utils.Cpuct * self.priorP * np.sqrt(self.parent_node.counter) / (1 + self.counter)


class node:
    def __init__(self, parent, player):
        self.parent = parent
        self.counter = 0.0
        self.child = {}
        self.node_player = player

    def add_child(self, action, priorP):  # 增加node之下的一个edge，但是没有实际创建新的node
        action_name = utils.move_to_str(action)
        self.child[action_name] = edge(action=action, parent_node=self, priorP=priorP)

    def get_child(self, action):
        child_node, _ = self.child[action].get_child()
        return child_node

    def eval_or_not(self):
        return len(self.child) == 0

    def backup(self, v):  # back propagation
        self.counter += 1
        if self.parent:
            self.parent.backup(v)

    def get_distribution(self, train=True):  ## used to get the step distribution of current
        for key in self.child.keys():
            distrib_calculater.push(key, self.child[key].counter)
        return distrib_calculater.get(train=train)

    #获取备选的指定数量下一步
    def get_distributions(self, train=True):  ## used to get the step distribution of current
        for key in self.child.keys():
            distrib_calculater.push(key, self.child[key].counter)
        return distrib_calculater.getChoiceMoves(train=train)

    def UCB_sim(self):  # 用于根据UCB公式选取node
        UCB_max = -sys.maxsize
        UCB_max_key = None
        for key in self.child.keys():
            if float(self.child[key].UCB_value().min()) > UCB_max:
                UCB_max_key = key
                UCB_max = self.child[key].UCB_value()  # 为什么UCB_value
        this_node, expand = self.child[UCB_max_key].get_child()
        return this_node, expand, self.child[UCB_max_key].action


class MCTS:
    def __init__(self, row_size, column_size, simulation_per_step=400, neural_network=None, matrixFile=None, diseaseScoreFile=None):
        self.row_size = row_size
        self.column_size = column_size
        self.s_per_step = simulation_per_step
        self.current_node = node(None, 1)
        self.NN = neural_network
        self.game_process = five_stone_game(row_size=row_size, column_size=column_size)  # 这里附加主游戏进程
        self.simulate_game = five_stone_game(row_size=row_size, column_size=column_size)  # 这里附加用于模拟的游戏进程

        # 加载矩阵文件
        matrixHead, disease, matrix = utils.read_excel(matrixFile)
        self.matrixHead = matrixHead
        self.matrix = matrix
        self.disease = disease

        # 加载疾病打分文件
        self.disease_symptom = {}
        with open(diseaseScoreFile, encoding='utf-8') as f:
            fs = ''.join(f)
            disease_symptom = json.loads(fs)
        self.disease_symptom = disease_symptom

        self.distribution_calculater = utils.distribution_calculater(self.row_size, self.column_size)

    def renew(self):
        self.current_node = node(None, 1)
        self.game_process.renew()

    def MCTS_step(self, action):
        next_node = self.current_node.get_child(action)
        next_node.parent = None
        return next_node

    def simulation(self):  # simulation的程序
        eval_counter, step_per_simulate = 0, 0
        for _ in range(self.s_per_step):
            expand, game_continue = False, True
            this_node = self.current_node
            self.simulate_game.simulate_reset(self.game_process.current_board_state(True))
            state = self.simulate_game.current_board_state()
            while game_continue and not expand:  # 当游戏未结束且无法扩展结点时
                if this_node.eval_or_not():
                    # self.NN.eval只有参数state，由utils.transfer_to_input给出
                    state_tmp = utils.transfer_to_input(state, self.simulate_game.which_player(), self.row_size, self.column_size)
                    state_prob, _ = self.NN.eval(state_tmp)
                    valid_move = utils.valid_move(state)
                    eval_counter += 1
                    for move in valid_move:
                        this_node.add_child(action=move, priorP=state_prob[0, move[0] * self.column_size + move[1]])

                this_node, expand, action = this_node.UCB_sim()
                game_continue, state = self.simulate_game.step(self, action)
                step_per_simulate += 1

            if not game_continue:  # 游戏结束
                this_node.backup(1)
            elif expand:  # 扩展结点
                _, state_v = self.NN.eval(
                    utils.transfer_to_input(state, self.simulate_game.which_player(), self.row_size, self.column_size))
                this_node.backup(state_v)
        return eval_counter / self.s_per_step, step_per_simulate / self.s_per_step

    def game(self, train=True):  # 主程序，产生game_record
        game_continue = True
        game_record = []
        begin_time = int(time.time())
        step = 1
        total_eval = 0
        total_step = 0
        # 模拟若干对局
        while game_continue:
            begin_time1 = int(time.time())
            avg_eval, avg_s_per_step = self.simulation()  # 更新UCB值中的参数
            action, distribution = self.current_node.get_distribution(train=train)  # 获得当前的action和distribution
            game_continue, state = self.game_process.step(self, utils.str_to_move(action))  # 获取执行action后的state和游戏是否结束的flag
            self.current_node = self.MCTS_step(action)  # 更新结点
            game_record.append({"distribution": distribution, "action": action})  # 保存一个时刻的对局信息
            end_time1 = int(time.time())
            print("step:{},cost:{}s, total time:{}:{} Avg eval:{}, Aver step:{}".format(step, end_time1 - begin_time1,
                                                                                        int(( end_time1 - begin_time) / 60),
                                                                                        (end_time1 - begin_time) % 60,
                                                                                        avg_eval, avg_s_per_step), end="\r")
            total_eval += avg_eval
            total_step += avg_s_per_step
            step += 1
        self.renew()
        end_time = int(time.time())
        min = int((end_time - begin_time) / 60)
        second = (end_time - begin_time) % 60
        print("In last game, we cost {}:{}".format(min, second), end="\n")  # 完成一次game的事件
        return game_record, total_eval / step, total_step / step

    def interact_game_init(self):
        self.renew()
        _, _ = self.simulation()
        action, distribution = self.current_node.get_distribution(train=False)
        game_continue, state = self.game_process.step(self, utils.str_to_move(action))
        self.current_node = self.MCTS_step(action)
        return state, game_continue

    def interact_game1(self, action):
        game_continue, state = self.game_process.step(self, action)
        return state, game_continue

    def interact_game2(self, action, game_continue, state):
        self.current_node = self.MCTS_step(utils.move_to_str(action))
        if not game_continue:
            pass
        else:
            _, _ = self.simulation()
            actions, distribution = self.current_node.get_distributions(train=False)

            for action in actions:
                place = utils.str_to_move(action["action"])
                val = self.matrix[place[0]][place[1]]
                cat = self.matrixHead[place[1]]
                action["val"] = val
                action["cat"] = cat
                action["grid"] = place
            action = actions[0]["action"]
            game_continue, state = self.game_process.step(self, utils.str_to_move(action))
            self.current_node = self.MCTS_step(action)
        return state, game_continue

    def interact_predict(self, action, game_continue, state):
        self.current_node = self.MCTS_step(utils.move_to_str(action))
        if not game_continue:
            pass
        else:
            _, _ = self.simulation()
            actions, distribution = self.current_node.get_distributions(train=False)
            result = []
            for action in actions:
                place = utils.str_to_move(action["action"])
                val = self.matrix[place[0]][place[1]]
                cat = self.matrixHead[place[1]]
                action["val"] = val
                action["cat"] = cat
                action["grid"] = place
                result.append(action)

        return actions
