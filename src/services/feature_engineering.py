"""
Funções para engenharia de features em dados diários de tickets
Para ser usado tanto no treinamento quanto na predição
"""
import pandas as pd
import numpy as np
from datetime import timedelta

# para usar no treinamento
def load_and_prepare(filepath_or_df):
    """Carrega e prepara dados diários de tickets"""
    if isinstance(filepath_or_df, str):
        df = pd.read_csv(filepath_or_df)
    else:
        df = filepath_or_df

    df['Date received'] = pd.to_datetime(df['Date received'], errors='coerce')
    df = df.dropna(subset=['Date received'])
    df['date'] = df['Date received'].dt.to_period('D').dt.to_timestamp()

    daily = df.groupby('date').size().rename('ticket_count').reset_index()
    daily = daily.sort_values('date').set_index('date').asfreq('D').fillna(0)
    daily.index.name = 'date'
    daily = daily.reset_index()

    daily['ticket_count'] = daily['ticket_count'].astype(int)

    return daily

# para usar tanto no treinamento
def create_time_features(df):
    """Cria features temporais para o modelo"""
    df = df.copy()

    # Features de data
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekday'] = df['date'].dt.weekday
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    # Lags (flags)
    flag = [1, 2, 3, 7, 14, 30]
    for l in flag:
        df[f'flag_{l}'] = df['ticket_count'].shift(l)

    # Rolling statistics
    windows = [3, 7, 14, 30]
    for w in windows:
        df[f'roll_mean_{w}'] = df['ticket_count'].shift(1).rolling(window=w, min_periods=1).mean()
        df[f'roll_std_{w}'] = df['ticket_count'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)

    # Diferenças e variações percentuais
    df['diff_1'] = df['ticket_count'].diff(1).fillna(0)
    df['pct_change_1'] = df['ticket_count'].pct_change(1).fillna(0).replace([np.inf, -np.inf], 0)

    # Flags de início e fim de mês
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    return df

# para usar na predição
def create_single_day_features(date, historical_data):
    """
    Cria features para um único dia futuro baseado em dados históricos
    
    Args:
        date: datetime - Data para criar features
        historical_data: DataFrame - Dados históricos com ticket_count
    
    Returns:
        dict - Dicionário com features para o dia
    """
    # Features básicas de data
    features = {
        'day': date.day,
        'month': date.month,
        'year': date.year,
        'weekday': date.weekday(),
        'weekofyear': date.isocalendar()[1],
        'is_month_start': 1 if date.day == 1 else 0,
        'is_month_end': 1 if date == (date + timedelta(days=1)).replace(day=1) - timedelta(days=1) else 0,
    }
    
    # Lags
    for lag in [1, 2, 3, 7, 14, 30]:
        idx = len(historical_data) - lag
        if idx >= 0 and idx < len(historical_data):
            features[f'flag_{lag}'] = historical_data.iloc[idx]['ticket_count']
        else:
            features[f'flag_{lag}'] = 0
    
    # Rolling stats
    for w in [3, 7, 14, 30]:
        recent_values = historical_data.tail(w)['ticket_count'].values
        features[f'roll_mean_{w}'] = np.mean(recent_values) if len(recent_values) > 0 else 0
        features[f'roll_std_{w}'] = np.std(recent_values) if len(recent_values) > 1 else 0
    
    # Diff e pct_change (placeholder para dia futuro)
    features['diff_1'] = 0
    features['pct_change_1'] = 0
    
    return features

# para usar na predição
def get_feature_columns():
    """Retorna lista ordenada de nomes das features (sem date e ticket_count)"""
    # Ordem exata das features geradas
    feature_names = [
        'day', 'month', 'year', 'weekday', 'weekofyear',
        'flag_1', 'flag_2', 'flag_3', 'flag_7', 'flag_14', 'flag_30',
        'roll_mean_3', 'roll_mean_7', 'roll_mean_14', 'roll_mean_30',
        'roll_std_3', 'roll_std_7', 'roll_std_14', 'roll_std_30',
        'diff_1', 'pct_change_1',
        'is_month_start', 'is_month_end'
    ]
    return feature_names

# para usar na predição
def prepare_features_for_prediction(feature_dict):
    """
    Prepara features para predição, garantindo ordem correta
    
    Args:
        feature_dict: dict - Dicionário com features
    
    Returns:
        DataFrame - Features na ordem correta
    """
    feature_names = get_feature_columns()
    # Garantir que todas as features existam
    ordered_features = {col: feature_dict.get(col, 0) for col in feature_names}
    return pd.DataFrame([ordered_features])[feature_names]