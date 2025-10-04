import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import torch


df = pd.read_csv("/content/train_news.csv")


texts = (df["title"].fillna("") + " " + df["publication"].fillna("")).tolist()

print(f"Загружено {len(texts)} текстов для обработки")


def get_local_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=32, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


    
    model = SentenceTransformer(model_name, device=device)

    
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Обрабатываю батч {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)
AVAILABLE_MODELS = {
    'small_fast': 'sentence-transformers/all-MiniLM-L6-v2',  
    'medium': 'sentence-transformers/all-mpnet-base-v2',     
    'multilingual': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  
    'large': 'sentence-transformers/all-distilroberta-v1'    
}



test_texts = texts[:5]

test_embeddings = get_local_embeddings(test_texts,
                                       model_name=AVAILABLE_MODELS['small_fast'],
                                       batch_size=2)

  

  
print(f"\nОбрабатываем все {len(texts)} текстов...")

selected_model = AVAILABLE_MODELS['small_fast']  

embeddings = get_local_embeddings(texts,
                                  model_name=selected_model,
                                  batch_size=64)



np.save("news_embeddings_local.npy", embeddings)

emb_df = pd.DataFrame(embeddings, index=df["Unnamed: 0"])
emb_df.to_csv("news_embeddings_local.csv")

print(f"- news_embeddings_local.npy")
print(f"- news_embeddings_local.csv")
