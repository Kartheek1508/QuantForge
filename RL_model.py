import pandas as pd
import random
import numpy as np
from data import reinforcement_data
import matplotlib.pyplot as plt

df = reinforcement_data("AAPL ","2020-01-01", "2025-01-01")
train_data = df[df.index<"2024-01-01"]
test_data = df[df.index >= "2024-01-01"]




class QAgent:

    def __init__(self):

        self.q_table = {}
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

        max_future_q = max(self.q_table[next_state].values())

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
        self.position_size = 0.1
        self.transaction_cost = 0.001
        returns = self.prices.pct_change()
        rolling_vol = returns.rolling(10).std()
        self.low_vol_thresh = rolling_vol.quantile(0.33)
        self.high_vol_thresh = rolling_vol.quantile(0.66)

    def reset(self):

        self.current_step = 0

        self.cash = self.initial_cash

        self.shares_held = 0

        self.portfolio_value = self.initial_cash

        return self.get_state()


    def get_state(self):

        if self.current_step < 10:
            momentum = "weak_up"
            vol_regime = "low_vol"
            ma_signal = "above_ma"

        else:
            current_price = self.prices.iloc[self.current_step]
            prev_price = self.prices.iloc[self.current_step - 1]

            returns = (current_price - prev_price) / prev_price

            if returns > 0.02:
                momentum = "strong_up"

            elif returns > 0:
                momentum = "weak_up"

            else:
                momentum = "down"

            recent_prices = self.prices.iloc[self.current_step - 10 : self.current_step]
            recent_returns = recent_prices.pct_change()
            volatility = recent_returns.std()

            ma_20 = recent_prices.mean()

            if current_price>ma_20:
                ma_signal = "above_ma"
            else:
                ma_signal = "below_ma"

            if volatility<self.low_vol_thresh:
                vol_regime = "low_vol"
            elif volatility<self.high_vol_thresh:
                vol_regime = "medium_vol"
            else:
                vol_regime = "high_vol"

        holding = self.shares_held > 0
        state = ( momentum,vol_regime, holding,ma_signal)

        return state



    def step(self, action):

        current_price = self.prices.iloc[self.current_step]

        old_portfolio_value = (self.portfolio_value )


        # BUY
        if action == 1:

            if (self.cash >= current_price and self.shares_held == 0):
                amount_invest = self.cash*self.position_size
                shares_to_buy = amount_invest/current_price
                fee = amount_invest*self.transaction_cost
                self.cash -= amount_invest+fee
                self.shares_held += shares_to_buy



        # SELL
        elif action == 2:

            if self.shares_held > 0:

                sale_val = self.shares_held*current_price
                fee = self.transaction_cost*sale_val
                self.cash += sale_val-fee

                self.shares_held = 0


        # HOLD
        else:
            pass

        self.current_step += 1


        if self.current_step >= len(self.prices) - 1:

            done = True

        else:

            done = False


        new_price = self.prices.iloc[self.current_step]

        self.portfolio_value = self.cash +(self.shares_held * new_price)
        reward = reward = (self.portfolio_value -old_portfolio_value) / old_portfolio_value

        next_state = self.get_state()

        return next_state, reward, done


agent = QAgent()

training_env = TradingEnv(train_data)
testing_env = TradingEnv(test_data)

for _ in range(1000):

    state = training_env.reset()

    done = False

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = training_env.step(action)

        agent.update_q_values(state,
            action,
            reward,
            next_state
        )

        state = next_state


    agent.epsilon *= 0.995


print(agent.q_table)

agent.epsilon = 0
portfolio_history = []
actions_history = []
price_history = []

state = testing_env.reset()

done = False

while not done:

    action = agent.choose_action(state)

    next_state, reward, done = testing_env.step(action)
    portfolio_history.append(testing_env.portfolio_value)
    actions_history.append(action)
    price_history.append(testing_env.prices.iloc[testing_env.current_step])
 
    state = next_state

initial_value = portfolio_history[0]

final_value = portfolio_history[-1]

total_return = ( final_value - initial_value) / initial_value

portfolio_returns = pd.Series(portfolio_history).pct_change().dropna()
sharpe_ratio = (portfolio_returns.mean()/portfolio_returns.std())
rolling_max = pd.Series(portfolio_history).cummax()

drawdown = (pd.Series(portfolio_history)- rolling_max) / rolling_max

max_drawdown = drawdown.min()

initial_price = test_data.iloc[0]

final_price = test_data.iloc[-1]

buy_hold_return = (final_price - initial_price) / initial_price

plt.plot(portfolio_history)

print("Total returns: ",total_return)
print("Sharpe ratio: ",sharpe_ratio)
print("Max drawdown: ",max_drawdown)
print("Buy and hold return: ",buy_hold_return)