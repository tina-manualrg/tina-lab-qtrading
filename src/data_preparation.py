import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import tags

def log_returns(prices: pd.Series, tau: int) -> pd.Series:
    """
    Compute log returns on time series data
    log(p2/p1) = log(p2) - log(p1)
    @param prices: time series data
    @param tau: differencing period
    @return: returns
    """
    return np.log(prices).diff(tau)

def factor_demean(data: pd.DataFrame, factor_col: str, sector_col='sector', date_col='date', return_frame=False):
    """
    On each period, substract mean factor value by sector to remove general sector effect from each single ticker (sector neutralization)
    """
    res = data.groupby([date_col, sector_col])[factor_col].transform(lambda grp: grp - grp.mean())
    if return_frame:
        return res.to_frame()
    else:
        return res

def factor_rank(data: pd.DataFrame, factor_col: str, date_col='date', return_frame=False):
    """
    On each period, rank to factor values, assigning 1 to the lowest factor value and N to the largest
    """
    res = data.groupby(date_col)[factor_col].rank()
    if return_frame:
        return res.to_frame()
    else:
        return res

def roll_zscore(x: pd.Series, window=20):
    mu = x.rolling(window).mean()
    sigma = x.rolling(window).std()
    return (x - mu) / sigma

def zscore(x: pd.Series):

    return (x - np.mean(x)) / np.std(x)

def factor_zscore(data: pd.DataFrame, factor_col: str, date_col='date', return_frame=False):
    """
    On each period, compute factor mean and std and standardize factor values: (x-mean)/std
    """
    res = data.groupby(date_col)[factor_col].transform(zscore)
    if return_frame:
        return res.to_frame()
    else:
        return res

def smooth(x, window: int):
    return x.rolling(window).mean()

def factor_smooth(data: pd.DataFrame, factor_col: str, ticker_col='ticker', window=20, return_frame=False):
    """
    Compute rolling average for each ticker factor values
    """
    res = data.groupby(ticker_col)[factor_col].transform(smooth, window=window)
    if return_frame:
        return res.to_frame()
    else:
        return res


def ohlc(open_col='adj_open', close_col='adj_close', high_col='adj_high', low_col='adj_low', rtol=0.01):

    # how can be this code speed up more?
    if (open_col <= close_col):
        flg_open_vs_low = np.isclose(open_col, low_col, rtol=rtol)
        flg_close_vs_high = np.isclose(close_col, high_col, rtol=rtol)
        if flg_open_vs_low and flg_close_vs_high:
            return 3
        elif flg_close_vs_high:
            return 2
        elif flg_open_vs_low:
            return 1
        else:
            return 0
    else:
        flg_close_vs_low = np.isclose(close_col, low_col, rtol=rtol)
        flg_open_vs_high = np.isclose(open_col, high_col, rtol=rtol)
        if flg_open_vs_high and flg_close_vs_low:
            return -3
        elif flg_close_vs_low:
            return -2
        elif flg_open_vs_high:
            return -1
        else:
            return 0


def macd(prices: pd.Series, win_short = 5, win_long=20):
    short_mave = prices.rolling(win_short).mean()
    long_mave = prices.rolling(win_long).mean()

    return short_mave - long_mave


def rsi(returns: pd.Series, window=14):
    mask_up = returns > 0
    data_idx = returns.index
    ups = pd.Series(index=data_idx, data=np.where(returns[mask_up], returns, np.nan), name='ups')
    downs = pd.Series(index=data_idx, data=np.where(-returns[~mask_up], returns, np.nan), name='downs')
    mean_up = ups.rolling(window).mean()
    mean_down = downs.rolling(window).mean()
    rs = mean_up / mean_down

    rsi = 100 - 100 / (1 + rs)
    rsi = np.where(mean_down == 0, 100, rsi)
    return rsi


def bollinger_bands(z_scores: pd.Series, threshold=2):
    return np.where(abs(z_scores) >= threshold, -1*np.sign(z_scores), 0)

def volatility(returns: pd.Series, window= 20):
    return returns.rolling(window).std()

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return ** 2))

def pl_targets(data: pd.DataFrame, tau_fwd: int, period = 'd') -> pd.DataFrame:
    """
    Target computation on close price
    @param data: input
    @param tau_fwd: target time horizon (forward)
    @return: output
    """
    target_col = f'y_fwd_logrets_{tau_fwd}{period}'
    return (data
            .assign(**{
                target_col: -1*data.groupby(tags.TICKER)[tags.CLOSE].apply(log_returns, -tau_fwd)})
            .dropna(subset=[target_col])
    )

def pl_min_avol_filter(data: pd.DataFrame, window: int, top_n: int) -> pd.DataFrame:
    """
    Compute rolling Average Volume by ticker and filter top quantile
    @param data: input
    @param window: number o periods to compute rolling mean
    @param top_n: filter by top n tickers (by period)
    @return: output
    """
    avol = data.groupby(tags.TICKER)[[tags.VOLUME]].transform(lambda grp: grp.rolling(window, min_periods=1).mean())
    avol['rank'] = avol.groupby(tags.DATE)[tags.VOLUME].rank(ascending=False)  #nlargest()
    mask_top_avol = avol['rank'] <= top_n

    return (data.loc[mask_top_avol])


def pl_date_filter(data: pd.DataFrame, start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Apply a filter to index level `date`
    @param data: input
    @param start_dt: Filter start
    @param end_dt: Filter end
    @return: output

    """
    mask_dates = (start_dt <= data.index.get_level_values(tags.DATE)) & (data.index.get_level_values(tags.DATE) <= end_dt)
    return (data.loc[mask_dates])


def pl_add_sector(data: pd.DataFrame, sector_map: pd.Series):

    current_idx = data.index.names
    data_wsector = (data
        .reset_index()
        .merge(sector_map, on=tags.TICKER, how='left')
        .set_index(current_idx)
    )

    data_wsector['sector'] = data_wsector['sector'].fillna('NA')

    return data_wsector

def pl_mkt_returns(data: pd.DataFrame, market_prices: pd.Series, tau: int):
    current_idx = data.index.names
    mkt_rets = log_returns(market_prices, tau=tau).fillna(0.)
    data_wmktrets = (data
                    .reset_index()
                    .merge(mkt_rets, on=tags.DATE, how='left')
                    .set_index(current_idx))
    return data_wmktrets

def pl_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features:
        x_returns_5d: 5 days momentum indicator
        x_returns_1y: 1 day momentum indicator
        x_ohlc_intraday: intraday OHLC indicator
    :param data: input
    :return:  output
    """
    data_grp = data.groupby(tags.TICKER)
    x_returns_5d = data_grp[tags.CLOSE].apply(log_returns, 5)
    time_idx = data.index.get_level_values(tags.DATE)
    x_price_zscore60d = data_grp[tags.CLOSE].transform(roll_zscore, window=60)

    features1 = data.assign(
        # intra-target term alphas
        x_intrad_upshadow=data.Adj_High - np.maximum(data.adj_close, data.adj_Open),
        x_intrad_loshadow=np.minimum(data.adj_Close, data.adj_Open) - data.adj_Low,
        x_returns_5d=x_returns_5d,
        x_zscore_5d=data_grp[tags.CLOSE].transform(roll_zscore, window=5),
        x_zscore_vol_5d=data_grp[tags.VOLUME].transform(roll_zscore, window=5),
        # x_ohlc_intraday=data[['adj_open', 'adj_close', 'adj_high', 'adj_low']].apply(lambda row: ohlc(*row), axis=1),

        # target-term alphas
        x_macd_5d_vs_20d=data_grp[tags.CLOSE].transform(macd, win_short=5, win_long=20),
        x_rsi=x_returns_5d.groupby(tags.TICKER).transform(rsi, window=14),
        x_zscore_20d=data_grp[tags.CLOSE].transform(roll_zscore, window=20),
        x_zscore_vol_20d=data_grp[tags.VOLUME].transform(roll_zscore, window=20),

        # long-tern alphas
        x_zscore_60d=x_price_zscore60d,
        x_zscore_vol_60d=data_grp[tags.VOLUME].transform(roll_zscore, window=60),
        x_bb_60d=bollinger_bands(x_price_zscore60d, threshold=2),
        x_macd_50d_vs_252d=data_grp[tags.CLOSE].transform(macd, win_short=50, win_long=252),
        x_returns_1y=data_grp[tags.CLOSE].pct_change(252),

        # risk
        x_rets5d_vol_60d=x_returns_5d.groupby(tags.TICKER).transform(volatility, window=60),
        x_rets5d_vol_120d=x_returns_5d.groupby(tags.TICKER).transform(volatility, window=120),
        x_mkt_dispersion=x_returns_5d.groupby(tags.DATE).std(),

        # date-features
        x_wday_cos=np.cos(time_idx.weekday * (2 * np.pi / 5)),
        x_wday_sin=np.sin(time_idx.weekday * (2 * np.pi / 5)),
        x_is_eoq=time_idx.is_quarter_end
        )

    features_mkt = data.groupby(tags.DATE)['SP500'].last().to_frame('mkt_returns')
    features_mkt['x_mkt_volat_60d'] = features_mkt['mkt_returns'].rolling(60).std()
    features_mkt['x_mkt_volat_120d'] = features_mkt['mkt_returns'].rolling(120).std()

    features = (features1
                 .reset_index()
                 .merge(features_mkt[['x_mkt_volat_60d', 'x_mkt_volat_120d']], on=tags.DATE, how='left')
                 )


    return features




