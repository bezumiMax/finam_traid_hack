import pandas as pd
import pickle
import numpy as np
from traiding_environment import TradingEnvironment
from BQL import BayesianRegression

def load_trained_model(model_path):
    with open(model_path, 'rb') as f:
        agent = pickle.load(f)
    return agent

def test_model_single(test_news, test_emb, model_path):
    agent = load_trained_model(model_path)
    test_env = TradingEnvironment(test_candles, test_news, test_emb, n_states=200)
    agent.exploration = False
    
    state = test_env.reset()
    predictions = []
    actual_changes = []
    
    done = False
    while not done:
        current_news_vectors = test_env.get_news_vectors(test_env.index)
        
        action_index, predicted_change = agent.choose_action(
            state, current_news_vectors, exploration=False
        )
        
        predictions.append(predicted_change)
        
        next_state, reward, done, actual_change = test_env.step(
            action_index, predicted_change
        )
        
        if actual_change is not None:
            actual_changes.append(actual_change)
        
        state = next_state
    
    return predictions, actual_changes

def test_model_ensemble(test_news, test_emb, model_paths, aggregation='mean'):
    """
    Тестирование ансамбля моделей с усреднением результатов
    Parameters:
    - model_paths: список путей к файлам моделей
    - aggregation: метод агрегации ('mean', 'median', 'weighted')
    """
    all_predictions = []
    actual_changes = None
    
    # Получаем предсказания от всех моделей
    for model_path in model_paths:
        print(f"Загрузка модели: {model_path}")
        predictions, actual_changes = test_model_single(test_candles, test_news, test_emb, model_path)
        all_predictions.append(predictions)
    
    # Преобразуем в numpy array для удобства вычислений
    all_predictions = np.array(all_predictions)
    
    # Усредняем предсказания
    if aggregation == 'mean':
        ensemble_predictions = np.mean(all_predictions, axis=0)
    elif aggregation == 'median':
        ensemble_predictions = np.median(all_predictions, axis=0)
    elif aggregation == 'weighted':
        # Простое взвешивание - можно настроить под ваши нужды
        weights = np.ones(len(model_paths)) / len(model_paths)
        ensemble_predictions = np.average(all_predictions, axis=0, weights=weights)
    else:
        raise ValueError("Неизвестный метод агрегации")
    
    return ensemble_predictions.tolist(), actual_changes, all_predictions

# Список всех моделей
model_paths = [
    'trained_agent_aflt.pkl',
    'trained_agent_alrs.pkl',
    'trained_agent_chmf.pkl',
    'trained_agent_gazp.pkl',
    'trained_agent_gmkn.pkl',
    'trained_agent_lkoh.pkl',
    'trained_agent_magn.pkl',
    'trained_agent_mgnt.pkl',
    'trained_agent_moex.pkl',
    'trained_agent_mtss.pkl',
    'trained_agent_nvtk.pkl',
    'trained_agent_phor.pkl',
    'trained_agent_plzl.pkl',
    'trained_agent_rosn.pkl',
    'trained_agent_rual.pkl',
    'trained_agent_sber.pkl',
    'trained_agent_sibn.pkl',
    'trained_agent_t.pkl',
    'trained_agent_vtbr.pkl'
]

# Загрузка тестовых данных
test_candles = pd.read_csv('S:/test_candles.csv')
test_news = pd.read_csv('S:/test_news.csv')
test_emb = pd.read_csv('S:/test_embeddings.csv')

test_news['publish_date'] = pd.to_datetime(test_news['publish_date'])
test_news['date_only'] = test_news['publish_date'].dt.date
aflt_test_candles = test_candles[test_candles['ticker'] == 'AFLT']

# Тестирование ансамбля моделей
ensemble_predictions, actual_changes, all_predictions = test_model_ensemble(
    aflt_test_candles, test_news, test_emb, model_paths, aggregation='mean'
)

print("Усредненные предсказания ансамбля:")
for i, pred in enumerate(ensemble_predictions):
    print(f"День {i+1}: {pred:.3f}%")

# Дополнительная информация о разбросе предсказаний
print(f"\nСтатистика по ансамблю:")
print(f"Количество моделей: {len(model_paths)}")
print(f"Среднее предсказание за первый день: {np.mean([pred[0] for pred in all_predictions]):.3f}%")
print(f"Стандартное отклонение: {np.std([pred[0] for pred in all_predictions]):.3f}%")
print(f"Мин-Макс: [{np.min([pred[0] for pred in all_predictions]):.3f}%, {np.max([pred[0] for pred in all_predictions]):.3f}%]")

# Сохранение результатов
results_df = pd.DataFrame({
    'ensemble_prediction': ensemble_predictions,
    'actual_change': actual_changes
})

# Добавляем предсказания отдельных моделей
for i, model_path in enumerate(model_paths):
    ticker_name = model_path.replace('trained_agent_', '').replace('.pkl', '')
    results_df[f'pred_{ticker_name}'] = all_predictions[i]

results_df.to_csv('ensemble_predictions.csv', index=False)
print("\nРезультаты сохранены в ensemble_predictions.csv")
