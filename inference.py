import os
from fastapi import FastAPI
from pydantic import BaseModel
from env.traffic_env import TrafficEnv

app = FastAPI()

env = TrafficEnv()

class Action(BaseModel):
    action: int

@app.post("/reset")
def reset():
    print("START")
    state = env.reset()
    print("END")
    return {"observation": list(state)}

@app.post("/step")
def step(action: Action):
    print("STEP")
    next_state, reward, done = env.step(action.action)
    return {
        "observation": list(next_state),
        "reward": float(reward),
        "done": bool(done)
    }

