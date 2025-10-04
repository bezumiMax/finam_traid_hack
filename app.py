import pandas as pd
from datetime import datetime

data_news = pd.read_csv('S:/train_news.csv')
emb_data_news = pd.read_csv('S:/news_embeddings_local.csv')
data_candles = pd.read_csv('S:/train_candles.csv')

data_news['publish_date'] = pd.to_datetime(data_news['publish_date'])
data_news['date_only'] = data_news['publish_date'].dt.date

aflt_data_candles = data_candles[data_candles['ticker'] == 'AFLT']
alrs_data_candles = data_candles[data_candles['ticker'] == 'ALRS']
chmf_data_candles = data_candles[data_candles['ticker'] == 'CHMF']
gazp_data_candles = data_candles[data_candles['ticker'] == 'GAZP']
gmkn_data_candles = data_candles[data_candles['ticker'] == 'GMKN']
lkoh_data_candles = data_candles[data_candles['ticker'] == 'LKOH']
magn_data_candles = data_candles[data_candles['ticker'] == 'MAGN']
mgnt_data_candles = data_candles[data_candles['ticker'] == 'MGNT']
moex_data_candles = data_candles[data_candles['ticker'] == 'MOEX']
mtss_data_candles = data_candles[data_candles['ticker'] == 'MTSS']
nvtk_data_candles = data_candles[data_candles['ticker'] == 'NVTK']
phor_data_candles = data_candles[data_candles['ticker'] == 'PHOR']
plzl_data_candles = data_candles[data_candles['ticker'] == 'PLZL']
rosn_data_candles = data_candles[data_candles['ticker'] == 'ROSN']
rual_data_candles = data_candles[data_candles['ticker'] == 'RUAL']
sber_data_candles = data_candles[data_candles['ticker'] == 'SBER']
sibn_data_candles = data_candles[data_candles['ticker'] == 'SIBN']
t_data_candles = data_candles[data_candles['ticker'] == 'T']
vtbr_data_candles = data_candles[data_candles['ticker'] == 'VTBR']
