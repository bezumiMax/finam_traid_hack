import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TradingEnvironment:
    def __init__(self, aflt_data_candles, data_news, emb_data_news, n_states=200):
        self.aflt_data_candles = aflt_data_candles
        self.data_news = data_news
        self.emb_data_news = emb_data_news
        self.n_states = n_states
        self.reset()

    def reset(self):
        self.index = 0
        self.position = 0
        self.cash = 10000
        self.portfolio_value = 10000
        return self.discretize_state(self.aflt_data_candles.iloc[0])

    def discretize_state(self, candle_data):
        """Преобразует рыночные данные в дискретное состояние"""
        price = candle_data['close']
        min_price = price - 10.0
        max_price = price + 10.0
        
        normalized = (price - min_price) / (max_price - min_price)
        state = int(normalized * (self.n_states - 1))
        return min(max(state, 0), self.n_states - 1)
    
    def get_news_vectors(self, current_index):
        if current_index >= len(self.aflt_data_candles) or current_index >= len(self.emb_data_news):
            return []
        embedding_row = self.emb_data_news.iloc[current_index]
        embedding = embedding_row.values.astype(float)
        return [embedding]
            
    
    def calculate_price_change(self, current_index):
        if current_index > len(self.aflt_data_candles) - 1:
            return 0.0
        
        cur_price = self.aflt_data_candles.iloc[current_index]['open']
        next_price = self.aflt_data_candles.iloc[current_index]['close']
        price_change = ((next_price - cur_price) / cur_price) * 100
        price_change = np.clip(price_change, -10.0, 10.0)
        
        return price_change
    
    def step(self, action_index, predicted_change):
        if self.index >= len(self.aflt_data_candles) - 1:
            next_state = self.discretize_state(self.aflt_data_candles.iloc[-1])
            return next_state, 0, True, None

        actual_price_change = self.calculate_price_change(self.index)
        prediction_error = abs(predicted_change - actual_price_change)
        reward = -np.exp(prediction_error / 5.0)  # Штрафуем за ошибку
        # Бонус за правильное направление
        if (predicted_change > 0 and actual_price_change > 0) or \
        (predicted_change < 0 and actual_price_change < 0):
            reward += 3.0  # Бонус за правильное направление
        else:
            reward -= 3.0  # Штраф за неправильное направление
        
        
        reward = np.clip(reward, -10.0, 10.0)
        
        self.index += 1
        done = self.index >= len(self.aflt_data_candles)
        
        next_state = self.discretize_state(self.aflt_data_candles.iloc[self.index])
        return next_state, reward, done, actual_price_change
