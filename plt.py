#深層強化学習のパラメータ更新の様子
import numpy as np
import torch
import pandas as pd
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
init_notebook_mode()
import datetime as dt
import plotly.io as pio
import matplotlib.pyplot as plt
import japanize_matplotlib



def plot_loss_reward(total_losses, total_rewards):
    figure = tools.make_subplots(rows=1, cols=2, subplot_titles=('loss', 'reward'), print_grid=False)
    figure.append_trace(Scatter(y=total_losses, mode='lines', line=dict(color='skyblue')), 1, 1)
    figure.append_trace(Scatter(y=total_rewards, mode='lines', line=dict(color='orange')), 1, 2)
    figure['layout']['xaxis1'].update(title='epoch')
    figure['layout']['xaxis2'].update(title='epoch')
    figure['layout'].update(height=400, width=900, showlegend=False)
    iplot(figure)

#訓練期間・テスト期間の取引状況の様子
def plot_train_test_by_q(train_env, test_env, Q, algorithm_name, date_split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def to_tensor(x):
      return torch.from_numpy(x.astype(np.float32)).to(device)
    def to_tensor_long(x):
      return torch.tensor([[x]], device=device, dtype=torch.long)
    import statistics
    # train
    pobs = train_env.reset()
    train_acts = []
    train_rewards = []
    train_value = []

    for _ in range(len(train_env.data)-1):
        input_state = np.array(pobs, dtype=np.float32)
        pact = Q(to_tensor(input_state))
        pact = np.argmax(pact.detach().cpu().numpy())

        if _ == len(train_env.data)-2:
          pact = 5
        obs, reward, done, pact = train_env.step(pact)
        train_acts.append(pact)
        train_rewards.append(reward)
        train_value.append(train_env.position_value_change)
        pobs = obs

    print(train_acts)
    print(train_rewards)
    print(train_value)
    train_profits = train_env.cash_in_hand
    train_trade_times = train_env.trade_times
    train_trade_win = train_env.trade_win

    pobs = test_env.reset()
    test_acts = []
    test_rewards = []
    test_value = []

    for _ in range(len(test_env.data)-1):
        input_state = np.array(pobs, dtype=np.float32)
        pact = Q(to_tensor(input_state))
        pact = np.argmax(pact.detach().cpu().numpy())
        if _ == len(test_env.data)-2:
          pact = 5
        obs, reward, done, pact = test_env.step(pact)
        test_acts.append(pact)
        test_rewards.append(reward)
        test_value.append(test_env.position_value_change)

        pobs = obs
    print(test_acts)
    print(test_rewards)
    print(test_value)
    test_profits = test_env.cash_in_hand
    test_trade_times = test_env.trade_times
    test_trade_win = test_env.trade_win

  
    # plot
    train_copy = train_env.data.copy()
    test_copy = test_env.data.copy()
    train_copy['act'] = train_acts + [np.nan]
    train_copy['reward'] = train_rewards + [np.nan]
    test_copy['act'] = test_acts + [np.nan]
    test_copy['reward'] = test_rewards + [np.nan]

    train_hold = train_copy[train_copy['act'] == 0]
    train_buy = train_copy[(train_copy['act'] == 1) | (train_copy['act'] == 2) | (train_copy['act'] == 3) | (train_copy['act'] == 4)]
    train_sell = train_copy[(train_copy['act'] == 5) | (train_copy['act'] == 6) | (train_copy['act'] == 7) | (train_copy['act'] == 8)]
    train_sell_special= train_copy[train_copy['act'] == 9]

    test_hold = test_copy[test_copy['act'] == 0]
    test_buy = test_copy[(test_copy['act'] == 1) | (test_copy['act'] == 2) | (test_copy['act'] == 3) | (test_copy['act'] == 4)]
    test_sell = test_copy[(test_copy['act'] == 5) | (test_copy['act'] == 6) | (test_copy['act'] == 7) | (test_copy['act'] == 8)]
    test_sell_special= test_copy[test_copy['act'] == 9]

    # train_hold = train_copy[train_copy['act'] == 0]
    # train_buy = train_copy[(train_copy['act'] == 1)]
    # train_sell = train_copy[(train_copy['act'] == 2)]
    # train_sell_special= train_copy[train_copy['act'] == 9]

    # test_hold = test_copy[test_copy['act'] == 0]
    # test_buy = test_copy[(test_copy['act'] == 1)]
    # test_sell = test_copy[(test_copy['act'] == 2)]
    # test_sell_special= test_copy[test_copy['act'] == 9]


    act_color0, act_color1, act_color2, act_color3 = 'gray', 'cyan', 'magenta', 'red'

    data = [
        Candlestick(x=train_hold.index, open=train_hold['Open'], high=train_hold['High'], low=train_hold['Low'], close=train_hold['Close'], increasing=dict(line=dict(color=act_color0)), decreasing=dict(line=dict(color=act_color0))),
        Candlestick(x=train_buy.index, open=train_buy['Open'], high=train_buy['High'], low=train_buy['Low'], close=train_buy['Close'], increasing=dict(line=dict(color=act_color1)), decreasing=dict(line=dict(color=act_color1))),
        Candlestick(x=train_sell.index, open=train_sell['Open'], high=train_sell['High'], low=train_sell['Low'], close=train_sell['Close'], increasing=dict(line=dict(color=act_color2)), decreasing=dict(line=dict(color=act_color2))),
        Candlestick(x=train_sell_special.index, open=train_sell_special['Open'], high=train_sell_special['High'], low=train_sell_special['Low'], close=train_sell_special['Close'], increasing=dict(line=dict(color=act_color3)), decreasing=dict(line=dict(color=act_color3))),


        Candlestick(x=test_hold.index, open=test_hold['Open'], high=test_hold['High'], low=test_hold['Low'], close=test_hold['Close'], increasing=dict(line=dict(color=act_color0)), decreasing=dict(line=dict(color=act_color0))),
        Candlestick(x=test_buy.index, open=test_buy['Open'], high=test_buy['High'], low=test_buy['Low'], close=test_buy['Close'], increasing=dict(line=dict(color=act_color1)), decreasing=dict(line=dict(color=act_color1))),
        Candlestick(x=test_sell.index, open=test_sell['Open'], high=test_sell['High'], low=test_sell['Low'], close=test_sell['Close'], increasing=dict(line=dict(color=act_color2)), decreasing=dict(line=dict(color=act_color2))),
        Candlestick(x=test_sell_special.index, open=test_sell_special['Open'], high=test_sell_special['High'], low=test_sell_special['Low'], close=test_sell_special['Close'], increasing=dict(line=dict(color=act_color3)), decreasing=dict(line=dict(color=act_color3))),

    ]
    title = '{}: train reward_mean {}, profits {}, train trade_times {}, train trade_win{}, test reward_mean {}, profits {}, test trade_times {}, test trade_win{}'.format(
        algorithm_name,
        (statistics.mean(train_rewards)*100).round(3),
        train_profits,
        train_trade_times,
        train_trade_win,
        (statistics.mean(test_rewards)*100).round(3),
        test_profits,
        test_trade_times,
        test_trade_win,
        # int(sum(train_rewards)),
        # int(train_profits),
        # int(sum(test_rewards)),
        # int(test_profits)
    )
    layout = {
        'title': title,
        'showlegend': False,
         'shapes': [
             {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper', 'line': {'color': 'rgb(0,0,0)', 'width': 1}}
         ],
        'annotations': [
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left', 'text': ' test data'},
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right', 'text': 'train data '}
        ]
    }
    figure = Figure(data=data, layout=layout)
    iplot(figure)