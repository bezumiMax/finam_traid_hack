import pandas as pd
import pickle
import numpy as np
import requests
import json
from datetime import datetime
from traiding_environment import TradingEnvironment
from BQL import BayesianRegression


API_KEY = "YOUR_API_KEY"  

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Trading Meta LLM",
    "Content-Type": "application/json"
}


def load_trained_model(model_path):
    with open(model_path, 'rb') as f:
        agent = pickle.load(f)
    return agent


def test_model_single(test_candles, test_news, test_emb, model_path):
    agent = load_trained_model(model_path)
    test_env = TradingEnvironment(test_candles, test_news, test_emb, n_states=200)
    agent.exploration = False

    state = test_env.reset()
    predictions = []
    actual_changes = []

    done = False
    while not done:
        current_news_vectors = test_env.get_news_vectors(test_env.index)
        action_index, predicted_change = agent.choose_action(state, current_news_vectors, exploration=False)
        predictions.append(predicted_change)

        next_state, reward, done, actual_change = test_env.step(action_index, predicted_change)
        if actual_change is not None:
            actual_changes.append(actual_change)

        state = next_state

    return predictions, actual_changes


def test_model_ensemble(test_candles, test_news, test_emb, model_paths, aggregation='mean'):
    all_predictions = []
    actual_changes = None

    for model_path in model_paths:
        print(f"🔹 Загрузка модели: {model_path}")
        predictions, actual_changes = test_model_single(test_candles, test_news, test_emb, model_path)
        all_predictions.append(predictions)

    all_predictions = np.array(all_predictions)

    if aggregation == 'mean':
        ensemble_predictions = np.mean(all_predictions, axis=0)
    elif aggregation == 'median':
        ensemble_predictions = np.median(all_predictions, axis=0)
    else:
        raise ValueError("Неизвестный метод агрегации")

    return ensemble_predictions.tolist(), actual_changes, all_predictions



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

test_candles = pd.read_csv('S:/test_candles.csv')
test_news = pd.read_csv('S:/test_news.csv')
test_emb = pd.read_csv('S:/test_embeddings.csv')

test_news['publish_date'] = pd.to_datetime(test_news['publish_date'])
test_news['date_only'] = test_news['publish_date'].dt.date

ticker = "AFLT"
aflt_test_candles = test_candles[test_candles['ticker'] == ticker]


ensemble_predictions, actual_changes, all_predictions = test_model_ensemble(
    aflt_test_candles, test_news, test_emb, model_paths, aggregation='mean'
)

print(f"\n Усреднённые предсказания ансамбля для {ticker}:")
print(ensemble_predictions[:10])


recent_prices = aflt_test_candles.tail(5).to_dict(orient='records')
recent_news = (
    test_news[test_news['ticker'] == ticker]
    .sort_values('publish_date', ascending=False)
    .head(5)
    [['title', 'text']]
    .to_dict(orient='records')
)

data_for_llm = {
    "ticker": ticker,
    "last_5_days": recent_prices,
    "ensemble_predictions": ensemble_predictions[-5:],
    "recent_news": recent_news,
    "timestamp": datetime.now().isoformat()
}

prompt = f"""
Ты — финансовый аналитик.
У тебя есть данные по компании {ticker} в формате JSON:

{json.dumps(data_for_llm, ensure_ascii=False, indent=2)}

На основе этих данных:
1. Предскажи следующую дневную доходность (примерно, как число)
2. Предскажи суммарную доходность за 20 торговых дней (примерно)
3. Вероятность роста на i-ом горизонте (в процентах)

Ответь строго в формате JSON:
{{
  "pred_return_1d": float,
  "pred_return_20d": float,
  "pred_prob_up_ih": float
}}
"""


data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are a helpful financial analysis assistant."},
        {"role": "user", "content": prompt}
    ]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    llm_response_text = result["choices"][0]["message"]["content"]

    print("\n Ответ от LLM:")
    print(llm_response_text)

    try:
        llm_json = json.loads(llm_response_text)
    except json.JSONDecodeError:
        print("Не удалось распарсить JSON от модели, сохраняю как текст.")
        llm_json = {"raw_response": llm_response_text}

    output_df = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "pred_return_1d": llm_json.get("pred_return_1d"),
        "pred_return_20d": llm_json.get("pred_return_20d"),
        "pred_prob_up_ih": llm_json.get("pred_prob_up_ih"),
        "raw_response": llm_response_text
    }])

    csv_filename = f"llm_predictions_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

    print(f"\n Результат сохранён в: {csv_filename}")

else:
    print(" Ошибка запроса:", response.status_code, response.text)
