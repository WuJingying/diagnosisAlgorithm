# fastapi入口
import uvicorn as u
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import mctsPredict

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'))
app.include_router(mctsPredict.MCTS_predict)
# http://127.0.0.1:8083/docs

if __name__ == "__main__":
    u.run(app=app, host="127.0.0.1", port=8083, workers=1)
