import pandas as pd


def parse_and_prep(csv_path: str, group_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    date_col = "Date received"
    if date_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"CSV precisa conter colunas '{date_col}' e '{group_col}'")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, group_col])
    df[group_col] = df[group_col].str.strip()
    df = df[[date_col, group_col]]
    df[date_col] = df[date_col].dt.normalize()
    return df


def make_daily_series(df: pd.DataFrame, group_value: str, group_col: str) -> pd.Series:
    sub = df[df[group_col] == group_value].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    date_col = "Date received"
    s = sub.groupby(date_col).size().rename("count")
    idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
    s = s.reindex(idx, fill_value=0)
    s.index.name = date_col
    return s


def create_lgb_features(
    series: pd.Series, lags=[1, 7, 14, 30], windows=[7, 30]
) -> pd.DataFrame:
    df = (
        pd.DataFrame(series).rename(columns={0: "y"})
        if isinstance(series, pd.Series)
        else series.copy()
    )
    df = (
        df.rename(columns={series.name: "y"})
        if series.name
        else df.rename(columns={0: "y"})
    )
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w).std().fillna(0)
    df["dayofweek"] = df.index.dayofweek
    df["day"] = df.index.day
    df["month"] = df.index.month
    df = df.dropna()
    return df


def _format_series_dict(d):
    """
    Converte um dict com chaves de datas para:
      - chave no formato 'AAAA/MM/DD'
      - valor como inteiro (round, >= 0)
    Ignora valores NaN.
    """
    out = {}
    for k, vv in (d or {}).items():
        if pd.isna(vv):
            continue
        # formatar a chave como data se possível
        try:
            dt = pd.to_datetime(k)
            key = dt.strftime("%Y-%m-%d")
        except Exception:
            key = str(k)
        out[key] = max(0, int(round(vv)))
    return out
