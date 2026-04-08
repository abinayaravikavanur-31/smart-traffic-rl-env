import numpy as np

class TrafficEnv:
    def __init__(self):
        self.max_cars = 20
        self.signal = 0  # 0 = North-South green, 1 = East-West green
        self.state = None

    def reset(self):
        # Initialize random cars in both directions
        self.state = np.random.randint(0, 10, size=2)  # [NS, EW]
        return self.state

    def step(self, action):
        """
        action:
        0 = keep same signal
        1 = switch signal
        """

        # Switch signal if action = 1
        if action == 1:
            self.signal = 1 - self.signal

        ns, ew = self.state

        # Cars movement logic
        if self.signal == 0:
            # North-South green
            ns = max(0, ns - 2)
            ew += np.random.randint(0, 3)
        else:
            # East-West green
            ew = max(0, ew - 2)
            ns += np.random.randint(0, 3)

        # Limit max cars
        ns = min(ns, self.max_cars)
        ew = min(ew, self.max_cars)

        self.state = np.array([ns, ew])

        # Reward: minimize waiting cars
        reward = -(ns + ew)

        done = False

        return self.state, reward, done
