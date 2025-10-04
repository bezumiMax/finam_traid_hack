from tqdm import tqdm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import torch

class BayesianRegression:
    def __init__(self, n_states, n_actions=200, news_embed_dim=384, alpha=0.1, gamma=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.news_embed_dim = news_embed_dim 
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = True
        
        self.prediction_range = np.linspace(-10.0, 10.0, n_actions)
        
        # Байесовские параметры для Q-значений
        self.q_means = np.zeros((n_states, n_actions))
        self.q_precisions = np.ones((n_states, n_actions)) * 0.1
        
        # Attention механизм для новостей
        self.state_query_weights = np.random.normal(0, 0.1, (n_states, news_embed_dim))
        self.news_key_weights = np.random.normal(0, 0.1, (news_embed_dim, news_embed_dim))
        self.news_value_weights = np.random.normal(0, 0.1, (news_embed_dim, n_actions))
        
        # Веса для объединения состояния и новостей
        self.fusion_weights = np.random.normal(0, 0.1, (news_embed_dim + n_states, n_actions))
        
        self.uncertainty = np.ones((n_states, n_actions))
        
        print(f"Инициализирован агент с news_embed_dim={news_embed_dim}")
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def get_action_index(self, prediction_value):
        """Получить индекс действия по численному значению предсказания"""
        return np.argmin(np.abs(self.prediction_range - prediction_value))
    
    def news_attention(self, state, news_vectors):
        """Attention механизм для новостей"""
        if len(news_vectors) == 0:
            return np.zeros(self.n_actions), np.array([])
        
        news_vectors = np.array(news_vectors)
        
        query = self.state_query_weights[state]  # [news_embed_dim]
        keys = np.tanh(np.dot(news_vectors, self.news_key_weights))  # [num_news, news_embed_dim]
        attention_scores = np.dot(keys, query)  # [num_news]
        attention_weights = self.softmax(attention_scores)  # [num_news]
        news_values = np.dot(news_vectors, self.news_value_weights)  # [num_news, n_actions]
        attended_news = np.sum(attention_weights[:, np.newaxis] * news_values, axis=0)  # [n_actions]
        
        return attended_news, attention_weights
    
    def get_q_value(self, state, news_vectors, action):
        """Q-значение теперь представляет ожидаемую доходность предсказания"""
        base_q = self.q_means[state, action]
        
        if len(news_vectors) > 0:
            attended_news, attention_weights = self.news_attention(state, news_vectors)
            news_effect = attended_news[action]
            combined_q = base_q + news_effect
            return combined_q, attention_weights
        else:
            return base_q, np.array([])
    
    def choose_action(self, state, new_vector, exploration=True):
        """Выбор действия-предсказания"""
        if exploration is None:
            exploration = self.exploration
        if exploration:
            # Thompson sampling для исследования
            sampled_q = np.zeros(self.n_actions)
            for action in range(self.n_actions):
                sampled_q[action] = np.random.normal(
                    self.q_means[state, action],
                    1.0 / np.sqrt(self.q_precisions[state, action] + 1e-8)
                )
            best_action = np.argmax(sampled_q)
        else:
            q_values = np.zeros(self.n_actions)
            for action in range(self.n_actions):
                q_values[action], _ = self.get_q_value(state, new_vector, action)
            best_action = np.argmax(q_values)
        
        prediction = self.prediction_range[best_action]
        return best_action, prediction
    
    def update(self, state, news_vectors, action, actual_price_change, next_state, next_news_vectors, done):
        predicted_change = self.prediction_range[action]
        if actual_price_change is None or not isinstance(actual_price_change, (int, float)):
            actual_price_change = 0.0
        prediction_error = abs(predicted_change - actual_price_change)
        reward = -np.exp(prediction_error / 5.0)
        if (predicted_change > 0 and actual_price_change > 0) or \
        (predicted_change < 0 and actual_price_change < 0):
            reward += 3.0
        else:
            reward -= 3.0
        reward = np.clip(reward, -10.0, 10.0)
        
        if done:
            target = reward
        else:
            next_action, _ = self.choose_action(next_state, next_news_vectors, exploration=False)
            next_q, _ = self.get_q_value(next_state, next_news_vectors, next_action)
            target = reward + self.gamma * next_q
        
        current_q, _ = self.get_q_value(state, news_vectors, action)
        
        # TD ошибка
        td_error = target - current_q
        
        # Обновляем Q-параметры
        self.q_means[state, action] += self.alpha * td_error
        self.q_precisions[state, action] += 1.0
        self.uncertainty[state, action] = 1.0 / self.q_precisions[state, action]
        
        return td_error, prediction_error
