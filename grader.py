from env.traffic_env import TrafficEnv

def evaluate():
    env = TrafficEnv()
    total_wait = 0

    state = env.reset()

    for _ in range(50):
        action = 0  # simple fixed signal
        state, reward, _ = env.step(action)
        total_wait += sum(state)

    avg_wait = total_wait / 50
    print("Average Waiting Time:", avg_wait)

if __name__ == "__main__":
    evaluate()
