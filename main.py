from env.traffic_env import TrafficEnv
from agent.q_learning import QLearningAgent

env = TrafficEnv()
agent = QLearningAgent()

for ep in range(50):
    state = env.reset()
    
    for step in range(30):
        action = agent.choose_action(state)
        next_state, reward, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state

    print("Episode:", ep)
