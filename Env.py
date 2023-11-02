class Environment:
    def __init__(self, data, pre_data, initial_money = 1000000, history_t=20):
        self.data = data
        self.pre_data = pre_data
        self.history_t = history_t
        self.initial_money = initial_money
        self.total_steps = len(self.data) -1
        self.close_data = [] #株価
        self.ful_money_data = []
        self.stock_array = []
        self.act_data = []
        self.profit_data = []
        self.sell_profit_data = []
        self.reward_data = []
        self.sell_price_data =[]
        self.sell_amount_data = []
        self.reset()


    def reset(self):
        self.t = 0
        self.end_step = self.total_steps
        self.done = False
        self.EMA5 = []
        self.EMA25 = []
        self.EMA75 = []
        self.MACD_SIGN = 0
        self.RSI9 = []
        self.RSI14 = []
        self.history = []
        self.volume = []
        for i in range(self.history_t):
          self.EMA5.append(self.pre_data.iloc[i, :]['5daysEMA_Normal'])
          self.EMA25.append(self.pre_data.iloc[i, :]['25daysEMA_Normal'])
          self.EMA75.append(self.pre_data.iloc[i, :]['75daysEMA_Normal'])
          self.RSI9.append(self.pre_data.iloc[i, :]['rsi9'])
          self.RSI14.append(self.pre_data.iloc[i, :]['rsi14'])
          self.history.append(self.pre_data.iloc[i, :]['Close_Normal'])
          self.volume.append(self.pre_data.iloc[i, :]['Volume_Normal'])

        self.cash_in_hand = self.initial_money #所持金
        self.position = 0 #保有株価
        self.positions_count = 0 #保有株数
        self.position_value = 0 #保有株の価値
        self.next_position_value = 0
        self.position_profit = 0 #保有株の利益
        self.all_money = self.cash_in_hand + self.position_value #総資産
        self.next_all_money = self.cash_in_hand + self.next_position_value

        self.position_stock = 0
        self.position_value_change = 0
        self.money_rate = 1
        self.stock_rate = 0

        self.trade_times = 0
        self.trade_win = 0

        return self.EMA5 + self.EMA25 + self.EMA75 + [self.MACD_SIGN] + self.RSI9 + self.RSI14 + self.history + self.volume + [self.position_stock] + [self.position_value_change] + [self.money_rate] + [self.stock_rate] #obs

    def step(self, act):
        reward = 0
        positions_count = 0
        self.done = (self.end_step == self.t)

        # if self.position_stock == 1 and self.position_value_change < -0.05: #ロスカットを考慮する際に追加
        #     act = 9

        if act == 1 or act == 2 or act == 3 or act == 4:
          positions_count = 0 #t期の購入株数
          if self.cash_in_hand <= 0:
            reward = 0
          else:
            if act == 1:
              positions_count = self.cash_in_hand // self.data.iloc[self.t, :]['Adj Close'] #最大株数
            if act == 2:
              positions_count = self.cash_in_hand // (self.data.iloc[self.t, :]['Adj Close'] * 2) #最大株数/2
            if act == 3:
              positions_count = self.cash_in_hand // (self.data.iloc[self.t, :]['Adj Close'] * 3) #最大株数/3
            if act == 4:
              positions_count = self.cash_in_hand // (self.data.iloc[self.t, :]['Adj Close'] * 4) #最大株数/4
            if positions_count == 0:
              reward = 0

            self.cash_in_hand -= self.data.iloc[self.t, :]['Adj Close'] * positions_count
            if (positions_count > 0 or self.positions_count > 0) and (positions_count + self.positions_count != 0):
                self.position = (self.position * self.positions_count + self.data.iloc[self.t, :]['Adj Close'] * positions_count) // (self.positions_count + positions_count) #平均保有株価
            self.positions_count += positions_count

        elif act == 5 or act == 6 or act == 7 or act == 8 or act == 9 or self.done: # sell
            if self.positions_count == 0 or self.position == 0:
              reward = 0
            else:
              positions_count = 0 #t期に売却する株数
              if act == 5 or act == 9 or self.done:
                positions_count = self.positions_count // 1 #最大株数
              if act == 6:
                positions_count = self.positions_count // 2 #最大株数/2
              if act == 7:
                positions_count = self.positions_count // 3 #最大株数/3
              if act == 8:
                positions_count = self.positions_count // 4 #最大株数/4
              if positions_count == 0:
                reward = 0
              self.cash_in_hand += self.data.iloc[self.t, :]['Adj Close'] * positions_count
              reward = ((self.data.iloc[self.t, :]['Adj Close'] - self.position) / self.position) * (positions_count / self.positions_count)

              if reward >0:
                self.trade_win += 1
              self.positions_count -= positions_count
        self.reward_data.append(reward)
        self.sell_price_data.append(self.position)
        self.sell_amount_data.append(positions_count)

        # set next time
        self.t += 1
        self.position_value = 0
        self.position_value = self.data.iloc[self.t - 1, :]['Adj Close'] * self.positions_count

        self.next_position_value = 0
        self.next_position_value = self.data.iloc[self.t, :]['Adj Close'] * self.positions_count

        self.position_profit=0
        self.position_profit = (self.data.iloc[self.t - 1, :]['Adj Close'] - self.position) * self.positions_count

        self.EMA5.pop(0)
        self.EMA5.append(self.data.iloc[self.t, :]['5daysEMA_Normal'])
        self.EMA25.pop(0)
        self.EMA25.append(self.data.iloc[self.t, :]['25daysEMA_Normal'])
        self.EMA75.pop(0)
        self.EMA75.append(self.data.iloc[self.t, :]['75daysEMA_Normal'])
        self.RSI9.pop(0)
        self.RSI9.append(self.data.iloc[self.t, :]['rsi9'])
        self.RSI14.pop(0)
        self.RSI14.append(self.data.iloc[self.t, :]['rsi14'])
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close_Normal'])
        self.volume.pop(0)
        self.volume.append(self.data.iloc[self.t, :]['Volume_Normal'])

        if self.data.iloc[self.t, :]['macd_sign'] == 'GC':
          self.MACD_SIGN = 1
        elif self.data.iloc[self.t, :]['macd_sign'] == 'DC':
          self.MACD_SIGN = -1
        else:
          self.MACD_SIGN = 0

        if self.positions_count > 0:
          self.position_stock = 1
        elif self.positions_count == 0:
          self.position_stock = 0

        if self.position_stock == 1:
           self.position_value_change = (self.data.iloc[self.t, :]['Adj Close'] - self.position) / self.position
        else:
          self.position_value_change = 0
        self.all_money = self.cash_in_hand + self.position_value
        self.next_all_money = self.cash_in_hand + self.next_position_value
        self.money_rate = self.cash_in_hand / self.next_all_money
        self.stock_rate = self.next_position_value / self.next_all_money

        #for graph

        self.close_data.append(self.data.iloc[self.t - 1, :]['Adj Close'])

        self.ful_money_data.append(self.all_money)
        self.profit_data.append(self.all_money - self.initial_money)

        if act == 5 or act == 6 or act == 7 or act == 8 or act == 9 or self.done:
            self.sell_profit_data.append((self.data.iloc[self.t - 1, :]['Adj Close'] - self.position) * positions_count)
        else:
            self.sell_profit_data.append(0)

        self.stock_array.append(self.positions_count)
        if act != 0 and positions_count ==0:
            self.act_data.append('hold')
        else:
            if act == 1 or act == 2 or act == 3 or act == 4:
                self.act_data.append('buy')
            elif act == 5 or act == 6 or act == 7 or act == 8 or self.done: # sell
                self.act_data.append('sell')
            elif act == 9:
                self.act_data.append('loss_cut')
            else:
                self.act_data.append('hold')

        if act != 0:
            self.trade_times += 1

        # print(self.position, self.positions_count, self.cash_in_hand, self.all_money)
        return self.EMA5 + self.EMA25 + self.EMA75 + [self.MACD_SIGN] + self.RSI9 + self.RSI14 + self.history + self.volume + [self.position_stock] + [self.position_value_change] + [self.money_rate] + [self.stock_rate], reward, self.done, act # obs, reward, done, act