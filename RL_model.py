import pandas as pd
import random
from data import monte_carlo_data

prices = monte_carlo_data(
    "AAPL",
    "2024-01-01",
    "2025-01-01"
)


class QAgent:

    def __init__(self):

        self.q_table = {
            ("up",False) : {
                0:0,
                1:0,
                2:0
            },
            ("up",True) : {
                0:0,
                1:0,
                2:0
            
            },
            ("down",False):{
                 0:0,
                1:0,
                2:0
            },
            ("down",True):{
                 0:0,
                1:0,
                2:0
            }
            
        }


        self.gamma = 0.99
        self.alpha = 0.1

        self.actions = [0, 1, 2]

        self.epsilon = 1


    def choose_action(self, state):

        if state not in self.q_table:

            self.q_table[state] = {
                0: 0,
                1: 0,
                2: 0
            }

        r = random.random()

        if r < self.epsilon:

            return random.choice(self.actions)

        else:

            return max(
                self.q_table[state].items(),
                key=lambda x: x[1]
            )[0]


    def update_q_values(
        self,
        state,
        action,
        reward,
        next_state
    ):

        if state not in self.q_table:

            self.q_table[state] = {
                0: 0,
                1: 0,
                2: 0
            }

        if next_state not in self.q_table:

            self.q_table[next_state] = {
                0: 0,
                1: 0,
                2: 0
            }

        current_q = self.q_table[state][action]

        max_future_q = max(
            self.q_table[next_state].values()
        )

        new_q = current_q + self.alpha * (
            reward +
            self.gamma * max_future_q -
            current_q
        )

        self.q_table[state][action] = new_q


class TradingEnv:

    def __init__(self, prices, initial_cash=10000):

        self.prices = prices

        self.initial_cash = initial_cash

        self.current_step = 0

        self.cash = initial_cash

        self.shares_held = 0

        self.portfolio_value = initial_cash


    def reset(self):

        self.current_step = 0

        self.cash = self.initial_cash

        self.shares_held = 0

        self.portfolio_value = self.initial_cash

        return self.get_state()


    def get_state(self):

        if self.current_step == 0:

            trend = "up"

        else:

            current_price = self.prices.iloc[
                self.current_step
            ]

            prev_price = self.prices.iloc[
                self.current_step - 1
            ]

            if current_price > prev_price:

                trend = "up"

            else:

                trend = "down"

        holding = self.shares_held > 0

        state = (trend, holding)

        return state


    def step(self, action):

        current_price = self.prices.iloc[
            self.current_step
        ]

        old_portfolio_value = (
            self.portfolio_value
        )


        # BUY
        if action == 1:

            if (
                self.cash >= current_price
                and self.shares_held == 0
            ):

                self.cash -= current_price

                self.shares_held += 1


        # SELL
        elif action == 2:

            if self.shares_held > 0:

                self.cash += current_price

                self.shares_held -= 1


        # HOLD
        else:
            pass


        self.current_step += 1


        if self.current_step >= len(self.prices) - 1:

            done = True

        else:

            done = False


        new_price = self.prices.iloc[
            self.current_step
        ]

        self.portfolio_value = (
            self.cash +
            (self.shares_held * new_price)
        )

        reward = (
            self.portfolio_value -
            old_portfolio_value
        )

        next_state = self.get_state()

        return next_state, reward, done


agent = QAgent()

env = TradingEnv(prices)


for _ in range(1000):

    state = env.reset()

    done = False

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        agent.update_q_values(state,
            action,
            reward,
            next_state
        )

        state = next_state


    agent.epsilon *= 0.995


print(agent.q_table)