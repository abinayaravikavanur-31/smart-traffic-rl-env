import os
from env.traffic_env import TrafficEnv

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

env = TrafficEnv()

def reset():
    print("START reset")

    state = env.reset()

    print("END reset")

    return {
        "observation": state.tolist()
    }

def step(action):
    print("STEP start")

    # Handle action format
    if isinstance(action, dict):
        action = action.get("action", 0)

    next_state, reward, done = env.step(action)

    print("STEP end")

    return {
        "observation": next_state.tolist(),
        "reward": float(reward),
        "done": bool(done)
    }
