import pandas as pd
from tqdm import tqdm
import numpy as np
from BQL import BayesianRegression
from traiding_environment import TradingEnvironment
from sklearn.model_selection import train_test_split
from app import (
    data_news, 
    emb_data_news, 
    data_candles, 
    aflt_data_candles, 
    alrs_data_candles,
    chmf_data_candles,
    gazp_data_candles,
    gmkn_data_candles,
    lkoh_data_candles,
    magn_data_candles,
    mgnt_data_candles,
    moex_data_candles,
    mtss_data_candles,
    nvtk_data_candles,
    phor_data_candles,
    plzl_data_candles,
    rosn_data_candles,
    rual_data_candles,
    sber_data_candles,
    sibn_data_candles,
    t_data_candles,
    vtbr_data_candles
)
import pickle


news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(aflt_data_candles, data_news, emb_data_news, n_states=200)
agent = BayesianRegression(
    n_states=200, 
    n_actions=200, 
    news_embed_dim=news_embed_dim
)


n_episodes = 100
rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        

        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")


with open('trained_agent_aflt.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_aflt.pkl")


news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(alrs_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_alrs.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_alrs.pkl")


news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(chmf_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")


with open('trained_agent_chmf.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_chmf.pkl")


news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(gazp_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_gazp.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_gazp.pkl")


news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(gmkn_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_gmkn.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_gmkn.pkl")


news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(lkoh_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_lkoh.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_lkoh.pkl")



news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(magn_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_magn.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_magn.pkl")



news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(mgnt_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_mgnt.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_mgnt.pkl")




news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(moex_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_moex.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_moex.pkl")


news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(mtss_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_mtss.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_mtss.pkl")



news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(nvtk_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_nvtk.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_nvtk.pkl")



news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(phor_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_phor.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_phor.pkl")



news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(plzl_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_plzl.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_plzl.pkl")




news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(rosn_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_rosn.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_rosn.pkl")



news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(rual_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_rual.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_rual.pkl")




news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(sber_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_sber.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_sber.pkl")



news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(sibn_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_sibn.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_sibn.pkl")




news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(t_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_t.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_t.pkl")



news_embed_dim = len(emb_data_news.columns)

env = TradingEnvironment(vtbr_data_candles, data_news, emb_data_news, n_states=200)

rewards_history = []
prediction_errors = []

for episode in tqdm(range(n_episodes)):
    state = env.reset()
    total_reward = 0
    total_prediction_error = 0
    steps = 0
    done = False
    
    while not done:
        current_news_vectors = env.get_news_vectors(env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=True)
        next_state, reward, done, actual_change = env.step(action_index, predicted_change)
        
        next_news_vectors = env.get_news_vectors(env.index)
        
        td_error, pred_error = agent.update(
            state, current_news_vectors, action_index, actual_change, 
            next_state, next_news_vectors, done
        )
        
        state = next_state
        total_reward += reward
        total_prediction_error += abs(pred_error)
        steps += 1
    
    avg_prediction_error = total_prediction_error / steps if steps > 0 else 0
    rewards_history.append(total_reward)
    prediction_errors.append(avg_prediction_error)
    
    if episode % 10 == 0:
        print(f"Эпизод {episode}, Награда: {total_reward:.2f}, Ошибка предсказания: {avg_prediction_error:.3f}")

# Анализ результатов
print(f"Средняя награда: {np.mean(rewards_history):.2f}")
print(f"Средняя ошибка предсказания: {np.mean(prediction_errors):.3f}")

with open('trained_agent_vtbr.pkl', 'wb') as f:
    pickle.dump(agent, f)

print("Модель сохранена в trained_agent_vtbr.pkl")
