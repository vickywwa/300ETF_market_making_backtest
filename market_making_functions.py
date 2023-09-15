import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import fsolve
import itertools
from decimal import Decimal, ROUND_UP, ROUND_DOWN


def custom_round(number, rounding_para):
    decimal_number = Decimal(str(number))
    rounded_number = float(decimal_number.quantize(
        Decimal('0.0001'), rounding=rounding_para))
    return rounded_number


def Black_scholes(option_type, S, K, T, sigma):
    d1 = (np.log(S/K)+0.5*T*sigma**2)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'C':
        price = S*norm.cdf(d1) - K*norm.cdf(d2)
    elif option_type == 'P':
        price = K*norm.cdf(-d2)-S*norm.cdf(-d1)
    return price


def get_implied_vol(option_type, S, K, T, option_price):
    calculate_price = lambda sigma: Black_scholes(option_type, S, K, T,sigma)-option_price
    sigma = newton(calculate_price, 0.15)
    # sigma = fsolve(calculate_price, x0=0.1)
    return sigma


def get_delta(option_type, S, K, T, sigma):
    d1 = (np.log(S/K)+0.5*T*sigma**2)/(sigma*np.sqrt(T))
    if option_type == 'C':
        delta = norm.cdf(d1)
    elif option_type == 'P':
        delta = norm.cdf(d1)-1
    return delta


def calc_vega(S, K, T, sigma):
    d1 = (np.log(S/K)+0.5*T*sigma**2)/(sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def calc_vol_adjusted(vega):
    # -1%Vega金额/4000*0.1%
    return -vega / 4000 * 0.001


def merge_price(df_ref, df_target):
    result = []
    temp = pd.merge(df_ref, df_target, on=['Time', 'exe_price'], how='left').sort_values(
        by=['exe_price', 'Time']).reset_index(drop=True)
    k_ref = df_ref['exe_price'].drop_duplicates()
    for k in k_ref:
        temp_temp = temp[temp['exe_price'] == k]
        temp_temp = temp_temp.fillna(method='ffill')
        result.append(temp_temp)
    return pd.concat(result, ignore_index=True).sort_values(by=['Time', 'exe_price']).reset_index(drop=True)


def modify_merge_price(fwd_price, diff_put_call, df_target):
    df_target['fwd_price'] = fwd_price
    df_target['diff_put_call'] = diff_put_call
    df_target['ref'] = df_target['Time'].astype(str).apply(lambda x:x[:-7]).astype(str).apply(lambda x:x[-5:])
    df_target = df_target[(df_target['ref']>='09:35')&(df_target['ref']<'14:55')]
    df_target = df_target[~((df_target['ref']>='11:28')&(df_target['ref']<'13:03'))].reset_index(drop=True)
    df_target = df_target.drop(['ref'],axis='columns')
    
    A = df_target.groupby('Time').apply(lambda x: x['diff_put_call'].idxmin())

    ATM = df_target['exe_price'][A].repeat(7).reset_index(drop=True)
    ATM_fwd = df_target['fwd_price'][A].repeat(7).reset_index(drop=True)

    df_target = df_target.iloc[pd.concat(
        [A-3, A-2, A-1, A, A+1, A+2, A+3]).sort_values()].reset_index(drop=True)

    df_target['ATM'] = ATM
    df_target['ATM_fwd'] = ATM_fwd

    df_target['log_price'] = np.log(
        df_target['exe_price']/df_target['ATM_fwd'])
    df_target['days_to_maturity'] = (pd.to_datetime(df_target['lasttradingdate'].astype(
        str))-pd.to_datetime(df_target['Time'].apply(lambda x: x[:-13]))).dt.days
    df_target['time_to_maturity'] = df_target['days_to_maturity']/365
    
    df_target[['TotalVol','SV1','BV1']] = df_target[['TotalVol','SV1','BV1']].astype(int)
    # df_target[['exe_price','ATM','log_price','ATM_fwd','time_to_maturity']] = df_target[['exe_price','ATM','log_price','ATM_fwd','time_to_maturity']].astype(np.float32)

    return df_target

# data preprocessing
def data_preprocessing(lasttradingdate, raw_data):
    Put_price = raw_data[(raw_data['lasttradingdate'] == lasttradingdate) & (
        raw_data['exe_mode'] == 'P')].reset_index(drop=True)
    Call_price = raw_data[(raw_data['lasttradingdate'] == lasttradingdate) & (
        raw_data['exe_mode'] == 'C')].reset_index(drop=True)
    Put_price.rename(columns={'mid_price': 'put_price'}, inplace=True)
    Call_price.rename(columns={'mid_price': 'call_price'}, inplace=True)

    time_list = raw_data['Time'].drop_duplicates().sort_values().reset_index(drop=True)
    strike_list = [3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4]

    df_ref = pd.DataFrame(itertools.product(time_list, strike_list), columns=['Time', 'exe_price'])

    put_merged_price = merge_price(df_ref, Put_price)
    call_merged_price = merge_price(df_ref, Call_price)

    put_merged_price.fillna(0, inplace=True)
    call_merged_price.fillna(0, inplace=True)

    fwd_price = call_merged_price['call_price'] + \
        put_merged_price['exe_price'] - put_merged_price['put_price']
    diff_put_call = abs(
        put_merged_price['put_price'] - call_merged_price['call_price'])

    put_data = modify_merge_price(fwd_price, diff_put_call, put_merged_price)
    call_data = modify_merge_price(fwd_price, diff_put_call, call_merged_price)
    
    put_data = put_data.drop(['fwd_price', 'diff_put_call','days_to_maturity'],axis='columns')
    call_data = call_data.drop(['fwd_price', 'diff_put_call','days_to_maturity'],axis='columns')
    
    return put_data, call_data


def solve_implied_vols_vega(put_data, call_data):
    otm_put = put_data[put_data.exe_price <= 4.2].copy()
    otm_put['vol'] = otm_put[otm_put.exe_price < otm_put.ATM].apply(lambda x: get_implied_vol(
        x.exe_mode, x.ATM_fwd, x.exe_price, x.time_to_maturity, x.put_price), axis=1)
    otm_put['BS_vega'] = otm_put[otm_put.exe_price < otm_put.ATM].apply(lambda x: calc_vega(
        x.ATM_fwd, x.exe_price, x.time_to_maturity, x.vol), axis=1)
    
    # otm_put['vol'] = otm_put.apply(lambda x: get_implied_vol(
    #     x.exe_mode, x.ATM_fwd, x.exe_price, x.time_to_maturity, x.put_price), axis=1)
    # otm_put['BS_vega'] = otm_put.apply(lambda x: calc_vega(
    #     x.ATM_fwd, x.exe_price, x.time_to_maturity, x.vol), axis=1)
    
    # otm_put[['vol','BS_vega']] = otm_put[['vol','BS_vega']].astype(np.float32)

    otm_call = call_data[call_data.exe_price >= 3.9].copy()
    otm_call['vol'] = otm_call[otm_call.exe_price >= otm_call.ATM].apply(lambda x: get_implied_vol(
        x.exe_mode, x.ATM_fwd, x.exe_price, x.time_to_maturity, x.call_price), axis=1)    
    otm_call['BS_vega'] = otm_call[otm_call.exe_price >= otm_call.ATM].apply(lambda x: calc_vega(
        x.ATM_fwd, x.exe_price, x.time_to_maturity, x.vol), axis=1)
    
    # otm_call['vol'] = otm_call.apply(lambda x: get_implied_vol(
    #     x.exe_mode, x.ATM_fwd, x.exe_price, x.time_to_maturity, x.call_price), axis=1)    
    # otm_call['BS_vega'] = otm_call.apply(lambda x: calc_vega(
    #     x.ATM_fwd, x.exe_price, x.time_to_maturity, x.vol), axis=1)
    
    # otm_call[['vol','BS_vega']] = otm_call[['vol','BS_vega']].astype(np.float32)

    return otm_put, otm_call

# polyfit, smoothed, and theo


def polyfit_vols(otm_put_price, otm_call_price):
    # df for polyfit with OTM Puts and Calls and ATM Calls
    df_for_polyfit = pd.concat([otm_put_price[(otm_put_price.exe_price < otm_put_price.ATM) & (otm_put_price.SP1 != otm_put_price.BP1)], otm_call_price[(otm_call_price.exe_price >= otm_call_price.ATM) & (otm_call_price.SP1 != otm_call_price.BP1)]],
                               ignore_index=True).sort_values(by=['Time', 'exe_price']).reset_index(drop=True)

    # polyfit --> [convex, skew, ATM vol fitted]
    fit_result = df_for_polyfit.groupby('Time').apply(lambda x: np.polyfit(x['log_price'], x['vol'], 2, w=np.sqrt(x['BS_vega']))).apply(
        pd.Series, index=['convex', 'skew', 'fit_ATM_vol']).reset_index()

    # new df with OTM and ATM Puts and Calls
    df_polyfit = pd.merge(df_for_polyfit, fit_result, on='Time', how='left')
    df_polyfit = pd.concat([df_polyfit, otm_put_price[otm_put_price.exe_price >= otm_put_price.ATM], otm_put_price[(otm_put_price.exe_price < otm_put_price.ATM) & (otm_put_price.SP1 == otm_put_price.BP1)], otm_call_price[otm_call_price.exe_price < otm_call_price.ATM], otm_call_price[(otm_call_price.exe_price >= otm_call_price.ATM) & (otm_call_price.SP1 == otm_call_price.BP1)]],
                           ignore_index=True).drop_duplicates(['Time', 'exe_price', 'exe_mode']).sort_values(by=['Time', 'fit_ATM_vol']).reset_index(drop=True)

    df_polyfit[['convex', 'skew', 'fit_ATM_vol']] = df_polyfit[[
        'convex', 'skew', 'fit_ATM_vol']].fillna(method='ffill')

    df_polyfit = df_polyfit.sort_values(by=['Time','exe_price','exe_mode'])

    # EWMA smooth ATM vols
    smoothed_ATM_vol = pd.DataFrame(
        df_polyfit[['Time', 'fit_ATM_vol']].drop_duplicates().reset_index(drop=True))
    smoothed_ATM_vol['smoothed_ATM_vol'] = smoothed_ATM_vol['fit_ATM_vol'].ewm(
        alpha=0.15, min_periods=60).mean()
    smoothed_vol = pd.merge(df_polyfit, smoothed_ATM_vol, on=[
                            'Time', 'fit_ATM_vol'], how='inner')
    smoothed_vol = smoothed_vol.sort_values(
        by=['Time', 'exe_price', 'exe_mode']).reset_index(drop=True)
    smoothed_vol['current_mid'] = smoothed_vol['call_price']
    smoothed_vol['current_mid'] = smoothed_vol['current_mid'].fillna(
        smoothed_vol['put_price'])
    smoothed_vol = smoothed_vol.drop(
        ['put_price', 'call_price'], axis='columns')
    smoothed_vol['smoothed_ATM_vol'] = smoothed_vol['smoothed_ATM_vol'].fillna(smoothed_vol['fit_ATM_vol'])

    ATM_min = smoothed_vol['ATM'].drop_duplicates().min()
    ATM_max = smoothed_vol['ATM'].drop_duplicates().max()

    smoothed_vol = smoothed_vol[(smoothed_vol['exe_price']>ATM_min-0.11)&(smoothed_vol['exe_price']<ATM_max+0.11)].reset_index(drop=True)

    return smoothed_vol


def calc_volume(bid, ask, cur_sp, cur_bp, cur_sv, cur_bv, cur_volu, next_volu):

    if bid >= cur_sp:
        buy_volume = min(10, cur_sv)
    elif (bid > cur_bp) and (bid < cur_sp):
        buy_volume = min(
            10, abs(next_volu - cur_volu)*0.3)
    elif (bid == cur_bp) and (cur_bv != 0):
        buy_volume = min(
            10, abs(next_volu - cur_volu)*0.3*10/cur_bv)
    else:
        buy_volume = 0

    if ask <= cur_bp:
        sell_volume = min(10, cur_bv)
    elif (ask > cur_bp) and (ask < cur_sp):
        sell_volume = min(
            10, abs(next_volu - cur_volu)*0.3)
    elif (ask == cur_sp) and (cur_sv != 0):
        sell_volume = min(
            10, abs(next_volu - cur_volu)*0.3*10/cur_sv)
    else:
        sell_volume = 0

    return round(buy_volume), round(sell_volume)



# 成交价格计算
def calc_deal_price(df):
    if df['current_position'] > 0:
        if df['bid'] > df['SP1']:
            return df['SP1']
        else:
            return df['bid']
    elif df['current_position'] < 0:
        if df['ask'] < df['BP1']:
            return df['BP1']
        else:
            return df['ask']
    else:
        return 0



def calc_price_and_volume(j, num, oneday_df, time_ref, vol_adjusted, tick_size, tick_adj, Rvol, cumulate_delta, cumulate_position, sub_cumulate_vega):
    current_df = oneday_df.loc[time_ref[j]]
    
    if j+1 <= len(time_ref)-1:
        next_df = oneday_df.loc[time_ref[j+1]]
    else:
        next_df = current_df

    # current_df['theo_vol'] = current_df['convex'] * current_df['log_price'] ** 2 + \
    #     current_df['skew'] * current_df['log_price'] + current_df[Rvol]

    theo_mid_price = current_df.apply(lambda x: Black_scholes(
        x.exe_mode, x.ATM_fwd, x.exe_price, x.time_to_maturity, x.vol + x[Rvol] - x.fit_ATM_vol + vol_adjusted), axis=1)
    # theo_mid_price = current_df.apply(lambda x: Black_scholes(
    #     x.exe_mode, x.ATM_fwd, x.exe_price, x.time_to_maturity, x.theo_vol + vol_adjusted), axis=1)
    current_df['bid'] = (theo_mid_price - 0.0001 * tick_size + tick_adj).apply(lambda x: custom_round(x, ROUND_DOWN))
    current_df['ask'] = (theo_mid_price + 0.0001 * tick_size + tick_adj).apply(lambda x: custom_round(x, ROUND_UP))

    cur_ref = current_df.reset_index(drop=True)
    A = cur_ref[cur_ref.exe_price == cur_ref.ATM].index
    
    buy_volume, sell_volume = np.zeros(num,dtype=int), np.zeros(num,dtype=int)
    cut_off_price = current_df[current_df.SP1 == current_df.BP1].exe_price.drop_duplicates().values
    trading_df = current_df.iloc[A[0]-1:A[1]+2]
    bid,ask,cur_sp,cur_bp,cur_sv,cur_bv,cur_volu,next_volu = trading_df['bid'].values, trading_df['ask'].values, trading_df['SP1'].values, trading_df['BP1'].values,trading_df['SV1'].values, trading_df['BV1'].values, trading_df['TotalVol'].values, next_df.iloc[A[0]-1:A[1]+2]['TotalVol'].values
    strike_list = trading_df.exe_price.values

    for m in range(len(current_df)):
        if cumulate_position[m]!=0:
            buy_volume[m], sell_volume[m] = calc_volume(current_df['bid'][m], current_df['ask'][m], current_df['SP1'][m],current_df['BP1'][m],current_df['SV1'][m],current_df['BV1'][m],current_df['TotalVol'][m],next_df['TotalVol'][m])
            if cumulate_position[m]>0:
                buy_volume[m] = 0
            elif cumulate_position[m]<0:
                sell_volume[m] = 0

    for k in range(4):
        if strike_list[k] not in cut_off_price:
            buy_volume[A[0]-1+k], sell_volume[A[0]-1+k] = calc_volume(bid[k], ask[k], cur_sp[k],cur_bp[k],cur_sv[k],cur_bv[k],cur_volu[k],next_volu[k])

    net_volume = buy_volume - sell_volume
    ATM = current_df.ATM[0]
    ATM_fwd = current_df.ATM_fwd[0]

    delta = current_df.apply(lambda x: get_delta(
        x.exe_mode, x.ATM_fwd, x.exe_price, x.time_to_maturity, x.vol), axis=1).values
    cash_delta = 10000 * net_volume * delta * ATM_fwd

    cumulate_delta += cash_delta.sum()
    hedge_volume = 0
    hedge_volume_list = np.zeros(num,dtype=int)
    if abs(cumulate_delta) >= 250000:
        hedge_volume = round(-cumulate_delta / (10000 * ATM_fwd))
        if ATM in cut_off_price:
            if ((ATM_fwd < ATM) & ((ATM-0.1) not in cut_off_price)) or ((ATM_fwd > ATM) & ((ATM+0.1) in cut_off_price)):
                hedge_volume_list[A[0]-2],hedge_volume_list[A[1]-2] = hedge_volume, -hedge_volume
                cumulate_delta += 10000 * ATM_fwd * (delta[A[0]-2]-delta[A[1]-2]) * hedge_volume
            elif ((ATM_fwd > ATM) & ((ATM+0.1) not in cut_off_price)) or ((ATM_fwd < ATM) & ((ATM-0.1) in cut_off_price)):
                hedge_volume_list[A[0]+2],hedge_volume_list[A[1]+2] = hedge_volume, -hedge_volume
                cumulate_delta += 10000 * ATM_fwd * (delta[A[0]+2]-delta[A[1]+2]) * hedge_volume
        else: 
            hedge_volume_list[A[0]],hedge_volume_list[A[1]] = hedge_volume, -hedge_volume
            cumulate_delta += 10000 * ATM_fwd * (delta[A[0]]-delta[A[1]]) * hedge_volume
        cash_delta = 10000 * (net_volume + hedge_volume_list) * delta * ATM_fwd
    
    current_position = net_volume + hedge_volume_list
    cumulate_position += current_position
    if abs(cumulate_position).sum() >= 3000:
        tick_adj = -0.0001 * cumulate_position/3000
    else: tick_adj = 0

    vega = calc_vega(current_df.ATM_fwd, current_df.exe_price, current_df.time_to_maturity, current_df.vol)
    cash_vega = 10000 * current_position * 0.01 * vega
    sub_cumulate_vega += cash_vega.sum()
    sub_vol_adjusted = calc_vol_adjusted(sub_cumulate_vega)

    current_df[['buy_volume','sell_volume', 'hedge_volume','current_position','cumulate_position','delta','cash_delta','cumulate_delta','vega','cash_vega','cumulate_vega','tick_adj','sub_vol_adjusted']
    ] = buy_volume, sell_volume, hedge_volume_list, current_position,cumulate_position, delta, cash_delta, cumulate_delta, vega, cash_vega, sub_cumulate_vega, tick_adj, sub_vol_adjusted
    
    return current_df, cumulate_delta, cumulate_position, tick_adj, sub_cumulate_vega, sub_vol_adjusted


def calc_ttl_vega(sub_cumulate_vega_1, sub_cumulate_vega_2, sub_cumulate_vega_3, sub_cumulate_vega_4):
    Tvega = 1/0.6 * sub_cumulate_vega_1 + 1/1 * sub_cumulate_vega_2 + 1/1.4 * sub_cumulate_vega_3 + 1/2.4 * sub_cumulate_vega_4
    T_vol_adjusted = calc_vol_adjusted(Tvega)
    return T_vol_adjusted

def calc_pnl(df):
    groups = df.groupby(['exe_price', 'exe_mode'])
    
    pnl_result = []
    for (exe_price, exe_mode), df_group in groups:
        df_group = df_group.sort_values(by='Time').reset_index(drop=True)
        cumulate_position = df_group['current_position'].cumsum()
        last_cumulate_position = cumulate_position.shift(fill_value=0)

        df_group['holding_pnl'] = last_cumulate_position * (df_group['current_mid'] - df_group['current_mid'].shift(fill_value=df_group['current_mid'].iloc[0]))
        df_group['trading_pnl'] = df_group['current_position'] * (df_group['current_mid'] - df_group.apply(calc_deal_price, axis=1))
        df_group['sub_ttl_pnl'] = df_group['holding_pnl'] + df_group['trading_pnl']
        
        pnl_result.append(df_group)

    pnl_result = pd.concat(pnl_result, ignore_index=True)
    pnl_result = pnl_result.sort_values(by=['Time', 'exe_price', 'exe_mode']).reset_index(drop=True)
    return pnl_result

def count_df(df):
    return df.groupby('Time').count().iloc[0,0]