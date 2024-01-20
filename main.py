import numpy as np
import pandas as pd
from get_data import get_data
from strategy import get_trades, get_oracle_trades
from backtest import backtest, backtest_baseline
import learners
import warnings
import datetime

FREQ = 1
FREQSTR = "1"
FLOATING = 0
START = 0
END = 0
GAMMA = 0.8


def DNE():
    df = get_data(0, 0, freq_per_second = FREQ, directory = "data_old/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    df = df.iloc[:-1]
    trades = get_trades(df)
    daily_values = backtest(df, trades, floating_cost=FLOATING)
    baseline_values = backtest_baseline(df, floating_cost=FLOATING)
    print("\nFinal Portfolio Value: " + str(daily_values.iloc[-1]))
    print("Net Value: " + str(daily_values.iloc[-1] - 200000))
    print("Cumulative Returns: " + str(daily_values.iloc[-1] / daily_values.iloc[0] - 1))
    print("Baseline Cumulative Returns: " + str(baseline_values.iloc[-1] / baseline_values.iloc[0] - 1))
    print("Baseline Net Value: " + str(baseline_values.iloc[-1] - 200000))

def ORACLE():
    df = get_data(0, 0, freq_per_second = FREQ, directory = "data_old/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    df = df.iloc[:-1]
    trades = get_oracle_trades(df)
    daily_values = backtest(df, trades, floating_cost=FLOATING)
    baseline_values = backtest_baseline(df, floating_cost=FLOATING)
    print("\nFinal Portfolio Value: " + str(daily_values.iloc[-1]))
    print("Net Value: " + str(daily_values.iloc[-1] - 200000))
    print("Cumulative Returns: " + str(daily_values.iloc[-1] / daily_values.iloc[0] - 1))
    print("Baseline Cumulative Returns: " + str(baseline_values.iloc[-1] / baseline_values.iloc[0] - 1))
    print("Baseline Net Value: " + str(baseline_values.iloc[-1] - 200000))


def RAW():
    df = get_data(0, 0, freq_per_second=FREQ, directory = "data_old/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    X = (df['b_size_0'] / df['a_size_0']).to_numpy().reshape(-1, 1)
    Y = (df['price'].shift(-1) / df['price']).to_numpy().reshape(-1, 1)
    Y[-1] = 0

    learner = learners.FFNNLearner(input_size=X[0].shape[0], output_size=Y[0].shape[0], hidden_sizes=[5, 5, 5, 5], data=df, learning_rate=0.00001, learning_decay=1)
    learner.train(X, Y, epochs=10000, verbose_freq=1)

    output = learner.test(X)
    output = np.where(output > 1, 1000, -1000)
    trades = pd.Series(output.squeeze(), index=df.index)
    trades.iloc[-1] = 0
    trades.iloc[1:] = trades.diff().iloc[1:]
    daily_values = backtest(df, trades)
    print(f"Net result: {round(daily_values.iloc[-1] - 200000, 2)}")

trip_times = []

def QL():
    # 1/60/61
    df = get_data(START, END, freq_per_second=FREQ, directory = "data_old/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    df = df.iloc[:-1] # size 23k
    print(df.shape)
    learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data=df, gamma=GAMMA, learning_rate=0.99, learning_decay=0.995, floating_cost=FLOATING)

    for trip in range(1501):
        start_time = datetime.datetime.now()
        learner.train(df)
        trades = pd.Series(0, index=df.index)
        for i in range(df.shape[0]):
            X = learner.test(i)
            trades.iloc[i] = (X - 1) * 1000
        trades.iloc[-1] = 0
        trades.iloc[1:] = trades.diff().iloc[1:]
        daily_values = backtest(df, trades, floating_cost=FLOATING)
        trip_times.append((datetime.datetime.now() - start_time).total_seconds())
        if trip % 1 == 0:
            print(f"Trip {trip} net result: {round(daily_values.iloc[-1] - 200000, 2)}")
            print("Learning Rate: ", round(learner.learning_rate, 5))
            print("Time Remaining: ", round(np.mean(trip_times) / 60 * (1500 - trip), 2))
        if trip % 10 == 0:
            learner.save_model(f"ql_model_{FREQSTR}.npy")

def run_model_ql(filename):
    df = get_data(0, 0, freq_per_second=FREQ, directory = "data_old/", orderbook_filename = "orderbook_1.csv", message_filename = "message_1.csv")
    df = df.iloc[:-1]
    learner = learners.LobsterLearner(input_size=2, output_size=3, hidden_sizes=[5, 5, 5, 5], data=df, gamma=0.1, learning_rate=0.99, learning_decay=0.995, floating_cost=FLOATING)
    learner.load_model(filename)
    trades = pd.Series(0, index=df.index)
    for i in range(df.shape[0]):
        X = learner.test(i)
        trades.iloc[i] = (X - 1) * 1000
    trades.iloc[-1] = 0
    trades.iloc[1:] = trades.diff().iloc[1:]
    daily_values = backtest(df, trades, floating_cost=FLOATING)
    print(f"\nNet result: {round(daily_values.iloc[-1] - 200000, 2)}")

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
QL()
DNE()
run_model_ql(f"ql_model_{FREQSTR}.npy")
# ORACLE()