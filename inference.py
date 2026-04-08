from env.traffic_env import TrafficEnv

env = TrafficEnv()

def reset():
    state = env.reset()
    return {"state": state.tolist()}

def step(action):
    next_state, reward, done = env.step(action)
    return {
        "state": next_state.tolist(),
        "reward": reward,
        "done": done
    }
