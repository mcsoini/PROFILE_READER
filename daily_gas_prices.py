#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import pandas as pd
from importlib import reload
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm
import numpy as np

import grimsel.auxiliary.timemap as timemap
import grimsel.auxiliary.aux_sql_func as aql

import PROFILE_READER.profile_reader as profile_reader

reload(profile_reader)

class DailyGasPriceReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='daily_gas_prices')
    data_dir = os.path.normpath('DAILY_FUELS/REUTER_GAS')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('hub', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('year', 'SMALLINT')]
    tb_pk = ['hub', 'year', 'hy']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read(self, fn):

        df_add = pd.read_excel(fn, header=[0,1,2]).dropna().reset_index()

        df_add = df_add[[c for c in df_add.columns if '€/MWh' in c or 'Date' in c]]

        df_add['DateTime'] = pd.to_datetime(df_add['Date'])
        df_add = df_add.drop('Date', axis=1)

        df_add.columns = df_add.columns.droplevel([-1, -2])

        df_add[df_add == '-'] = np.nan

        df_time_map = pd.DataFrame(index=pd.date_range(df_add.DateTime.min(),
                                                       df_add.DateTime.max(),
                                                       freq='D'))
        df_time_map = df_time_map.reset_index().rename(columns={'index':
                                                                'DateTime'})

        df_add = df_time_map.join(df_add.set_index('DateTime'), on='DateTime')

        df_add = df_add.set_index('DateTime')

        df_add = df_add.fillna(np.nan)

        df_add = df_add.loc[-df_add.isnull().all(axis=1)]

        df_add = (df_add.stack().reset_index()
                        .rename(columns={'level_1': 'hub', 0: 'value'}))


        df_add = df_add.pivot_table(index='DateTime', values='value', aggfunc=np.mean).reset_index()
        df_add['hub'] = 'mean_3_hubs'

        df_add['DateTime'] = df_add.DateTime.dt.tz_localize('UTC')

        tm = timemap.TimeMap(keep_datetime=True)
        tm.gen_hoy_timemap(freq='D', start=df_add.DateTime.min(), stop=df_add.DateTime.max())
        df_tm = tm.df_time_map.copy()

        df_tm['DateTime'] = pd.to_datetime(df_tm.DateTime.dt.date)
        df_tm['DateTime'] = df_tm.DateTime.dt.tz_localize('UTC')

        df_add_1 = pd.merge(df_tm, df_add, on='DateTime', how='left')

        df_add_1 = df_add_1.sort_values('DateTime')

        df_add_1['value'] = df_add_1['value'].fillna(method='ffill')
        df_add_1['hub'] = df_add_1['hub'].fillna(method='ffill')


        return df_add_1[['DateTime', 'value', 'hub']]


class QuandlCoalPriceReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='daily_coal_prices')
    data_dir = os.path.normpath('DAILY_FUELS/QUANDL_ROTTERDAM')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('index', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('mt_id', 'SMALLINT'),
               ('year', 'SMALLINT')]
    tb_pk = ['index', 'year', 'hy']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read(self, fn):

        df_add = pd.read_csv(fn)

        df_add['DateTime'] = pd.to_datetime(df_add.Date)
        df_add = (df_add.set_index('DateTime')['Settle']
                        .rename('value').reset_index())

        df_add['index'] = (fn.split(os.path.sep)[-1]
                             .split('_')[-1]
                             .split('.')[0])
        df_add['DateTime'] = df_add.DateTime.dt.tz_localize('UTC')

        return df_add

    def post_processing(self, df):
        df['mt_id'] = df.DateTime.dt.month - 1
        return df


if __name__ == '__main__':
    sys.exit()

    kw_dict = dict(dict_sql=dict(db='storage2'),
                   exclude_substrings=[],
                   tm_filt={'year': range(2005, 2018)},
                   ext=['csv'])


    op = QuandlCoalPriceReader(kw_dict)
    self = op
    fn = self.fn_list[0]
    op.read_all()



    kw_dict = dict(dict_sql=dict(db='storage2'),
                   exclude_substrings=[],
                   tm_filt={'year': range(2005, 2018)},
                   ext=['xlsx'])

    op = DailyGasPriceReader(kw_dict)
    self = op
    fn = self.fn_list[0]
    op.read_all()



# %%

