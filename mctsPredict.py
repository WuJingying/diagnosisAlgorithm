# fastapi入口
import torch
from fastapi import APIRouter

import utils

torch.set_default_tensor_type(torch.FloatTensor)
from new_MCTS import MCTS
import argparse
import warnings

warnings.filterwarnings('ignore')

from pydantic import BaseModel
import numpy as np
import json


class PredictModel(BaseModel):  # 自定义模型名字
    matrixFile = '0222矩阵.xlsx'
    diseaseScoreFile = 'symptomScores.txt'
    input = {"人群": "青少年人群", "症状": "三多一少"}
    model = 'model_200.pkl'  # 模型


MCTS_predict = APIRouter()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


@MCTS_predict.post('/MCTS_predict/')
async def mctsPredict(model: PredictModel):
    matrixFile = model.matrixFile
    diseaseScoreFile = model.diseaseScoreFile
    input = model.input
    modelName = model.model
    actions = predict(matrixFile, diseaseScoreFile, input, modelName)
    actions = np.asarray(actions)
    result = json.dumps(actions.tolist(), indent=4, cls=NpEncoder, ensure_ascii=False)
    print(result)

    return result


def predict(matrixFile, diseaseScoreFile, input, modelName):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="game", type=str,
                        help="decide which mode too choose, we can choose:display, game")
    parser.add_argument("--display_file", type=str, help="if we choose display mode, we must select a file to display")
    parser.add_argument("--game_model", type=str, help="if we choose game mode, we must select a trained model to use")
    args = parser.parse_args()

    args.mode = "game"  # 可选择game或display
    args.game_model = modelName  # game模式的模型文件

    if args.mode == "game":
        oppo = args.game_model
        record_file = None
    else:
        raise KeyError("we must select the 'game' mode.")

    if oppo:
        try:
            Net = torch.load(oppo, map_location='cpu')
            tree = MCTS(row_size=utils.row_size, column_size=utils.column_size, neural_network=Net,
                        matrixFile=matrixFile, diseaseScoreFile=diseaseScoreFile)
        except:
            raise ValueError("The parameter oppo must be a pretrained model")

    # 初始化棋盘
    tree.renew()
    _, _ = tree.simulation()
    m = np.matrix(tree.matrix)
    for key, value in input.items():
        grid = np.argwhere(m == value)[0]
        currentGrid = []
        currentGrid.append(grid[0])
        currentGrid.append(grid[1])
        movestr = utils.move_to_str(grid)
        grid[0] = grid[0] - 4
        grid[1] = grid[1] - 4
        record, game_continue = tree.interact_game1(grid)

    actions = tree.interact_predict(currentGrid, game_continue, record)

    return actions


if __name__ == "__main__":
    matrixFile = '0222矩阵.xlsx'
    diseaseScoreFile = 'symptomScores.txt'
    input = {"人群": "青少年人群", "症状": "三多一少"}
    predict(matrixFile, diseaseScoreFile, input)
