#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 16:50:56 2018

@author: user
"""

import sys, os
import pandas as pd
from importlib import reload
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import numpy as np
import itertools
import shutil

import grimsel_h.auxiliary.timemap as timemap
import grimsel_h.auxiliary.aux_sql_func as aql
from grimsel_h.auxiliary.aux_general import print_full

import PROFILE_READER.profile_reader as profile_reader

reload(profile_reader)

reload(profile_reader)


class TSOReader(profile_reader.ProfileReader):
    '''
    Some common methods.
    '''
    def get_fn_list(self, lst_res=['Wind', 'Solar']):
        '''
        Download the files from constructed urls. Append fn to fn_list

        Parameters:
        lst_res -- list of strings, names of resources as used to compose
                   the url
        '''

        # get complete list of urls
        lst_year = range(2005, 2020)
        lst_month = range(1, 13)
        lst_res = lst_res
        url_list = [self.url_base.format(mt=mt, yr=yr, res=res)
                    for yr, mt, res
                    in itertools.product(lst_year, lst_month, lst_res)]

        self.fn_list = []

        url = url_list[0]
        for url in url_list:

            pt = 'WIN_TOT' if 'wind' in url.lower() else 'SOL_PHO'

            s = url
            fn = '{}_{}_{}.csv'.format(self._dir.split(os.path.sep)[-1],
                                       pt, s[-5:].replace('-', '_'))
            _fn = os.path.join(self._dir, fn)
            self.fn_list.append(_fn)

            print('Downloading ' + s[:int(60/3 - 3)]
                  + '...'
                  + s[int(len(s)/2 - 10): int(len(s)/2 + 10)]
                  + '...'
                  + s[int(len(s) - 60/3 - 3):], end=' --- ')

            if not os.path.exists(_fn):

                r = requests.get(url)
                if r.status_code == 200:
                    with open(_fn, 'wb') as f:
                        f.write(r.content)

                    if not 'no data available' in next(r.iter_lines()).decode('utf-8'):
                        print('success.')
                    else:
                        print('success (no data available)')
                else:
                    print('failed (status code={}).'.format(r.status_code))
            else:
                print('skipping (file exists).')



class TennetReader(TSOReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='german_tso_tennet')
    data_dir = os.path.normpath('GERMAN_TSO/TENNET')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('val_type', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('tso', 'VARCHAR'),
               ('pp_id', 'VARCHAR'),
               ('year', 'SMALLINT')]
    tb_pk = ['val_type', 'year', 'hy', 'pp_id']

    exclude_substrings=[]

    url_base = ('http://www.tennettso.de/site/en/phpbridge?commandpath=Tats'
                + 'aechliche_und_prognostizierte_{res}energieeinspeisung%2Fm'
                + 'onthDataSheetCsv.php&sub=total&querystring=monat%3D'
                + '{yr:02d}-{mt:02d}')

    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list(lst_res=['Wind', 'Solar'])


    def read(self, fn):

        try:
            df_add = pd.read_csv(fn, delimiter=';', skiprows=3, index_col=False)
        except pd.errors.EmptyDataError as e:
            print(str(e))
            return None

        df_add = df_add.dropna(how='all', axis=1)

        df_add['Date'] = df_add['Date'].fillna(method='ffill')

        df_add['Position'] -= 1

        df_add['hour'] = np.floor(df_add.Position / 4).apply(int)
        df_add['minute'] = (np.floor(df_add.Position % 4) * 15).apply(int)

        for idt, dt in enumerate(['year', 'month', 'day']):
            df_add[dt] = df_add.Date.apply(lambda x: int(x.split('-')[idt]))

        df_add['DateTime'] = pd.to_datetime(df_add[['year', 'month', 'day',
                                                    'hour', 'minute']])
        lst_datacols = (['Forecast [MW]', 'Actual [MW]']
                        + (['Offshore contribution [MW]']
                            if 'WIN' in fn else []))

        df_add = (df_add.set_index('DateTime')[lst_datacols]
                        .stack().reset_index()
                        .rename(columns={'level_1': 'val_type', 0: 'value'}))

        df_add['val_type'] = (df_add.val_type
                                    .apply(lambda x: x.lower().replace(' [mw]', '')))

        df_add['pp_id'] = 'DE_' + '_'.join(fn.split(os.path.sep)[-1].split('_')[1:3])
        df_add['tso'] = 'tennet'

        df_add = self.time_resample(df_add)


        return df_add[['DateTime', 'tso', 'val_type', 'pp_id', 'value']]


    def postprocessing_tot(self):
        '''
        Various operations once the table df_tot has been assembled.
        '''
        self.tz_localize_convert(tz='UTC')
        self.df_tot = self.get_hour_of_the_year(self.df_tot)
        self.append_to_sql(self.df_tot.copy())


class AmprionReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='german_tso_amprion')
    data_dir = os.path.normpath('GERMAN_TSO/AMPRION')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('val_type', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('tso', 'VARCHAR'),
               ('pp_id', 'VARCHAR'),
               ('year', 'SMALLINT')]
    tb_pk = ['val_type', 'year', 'hy', 'pp_id']

    exclude_substrings=[]

    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        super().get_fn_list()


    def read(self, fn):

        df_add = pd.read_csv(fn, delimiter=';')

        df_add.columns = ['Date', 'Time', 'forecast_8am', 'total_estimate']


        df_add['Time'] = df_add.Time.apply(lambda x: x.split(' - ')[0])
        df_add['minute'] = df_add.Time.apply(lambda x: int(x[3:]))
        df_add['hour'] = df_add.Time.apply(lambda x: int(x[:2]))

        for idt, dt in enumerate(['day', 'month', 'year']):
            df_add[dt] = df_add.Date.apply(lambda x: int(x.split('.')[idt]))

        df_add['DateTime'] = pd.to_datetime(df_add[['year', 'minute', 'month', 'day', 'hour']])

        df_add = df_add.set_index('DateTime')[['forecast_8am', 'total_estimate']]

        df_add = df_add.applymap(lambda x: float(x.replace(',', ''))
                                           if type(x) is str else x)

        df_add = (df_add.stack().reset_index()
                        .rename(columns={'level_1': 'val_type', 0: 'value'}))


        df_add['pp_id'] = fn.split(os.path.sep)[-1][:2]
        dict_pp = {'pv': 'DE_SOL_PHO', 'wi': 'DE_WIN_ONS'}
        df_add['pp_id'] = df_add.pp_id.replace(dict_pp)

        df_add['tso'] = 'amprion'


        return df_add


    def postprocessing_tot(self):
        '''
        Various operations once the table df_tot has been assembled.
        '''
        self.tz_localize_convert(tz='UTC')
        self.df_tot = self.time_resample(self.df_tot)
        self.df_tot = self.get_hour_of_the_year(self.df_tot)
        self.append_to_sql(self.df_tot.copy())




class TransnetBWReader(TSOReader, profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='german_tso_transnetbw')
    data_dir = os.path.normpath('GERMAN_TSO/TRANSNETBW')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('val_type', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('tso', 'VARCHAR'),
               ('pp_id', 'VARCHAR'),
               ('year', 'SMALLINT')]
    tb_pk = ['val_type', 'year', 'hy', 'pp_id']

    exclude_substrings=[]

    url_base = ('https://api.transnetbw.de/{res}/csv?language='
                + 'en&date={yr:02d}-{mt:02d}')

    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list(lst_res=['wind', 'photovoltaics'])


    def read(self, fn):

        try:
            df_add = pd.read_csv(fn, delimiter=';', index_col=False)
        except pd.errors.EmptyDataError as e:
            print(str(e))
            return None

        df_add = df_add.dropna(how='all', axis=1)

        df_add['Date'] = df_add['Date'].fillna(method='ffill')

        df_add['Position'] -= 1

        df_add['hour'] = np.floor(df_add.Position / 4).apply(int)
        df_add['minute'] = (np.floor(df_add.Position % 4) * 15).apply(int)

        for idt, dt in enumerate(['year', 'month', 'day']):
            df_add[dt] = df_add.Date.apply(lambda x: int(x.split('-')[idt]))

        df_add['DateTime'] = pd.to_datetime(df_add[['year', 'month', 'day',
                                                    'hour', 'minute']])
        lst_datacols = (['Forecast [MW]', 'Actual [MW]']
                        + (['Offshore contribution [MW]']
                            if 'WIN' in fn else []))

        df_add = (df_add.set_index('DateTime')[lst_datacols]
                        .stack().reset_index()
                        .rename(columns={'level_1': 'val_type', 0: 'value'}))

        df_add['val_type'] = (df_add.val_type
                                    .apply(lambda x: x.lower().replace(' [mw]', '')))

        df_add['pp_id'] = 'DE_' + '_'.join(fn.split(os.path.sep)[-1].split('_')[1:3])
        df_add['tso'] = 'tennet'

        df_add = self.time_resample(df_add)


        return df_add[['DateTime', 'val_type', 'pp_id', 'value']]


    def postprocessing_tot(self):
        '''
        Various operations once the table df_tot has been assembled.
        '''
        self.tz_localize_convert(tz='UTC')
        self.df_tot = self.get_hour_of_the_year(self.df_tot)
        self.append_to_sql(self.df_tot.copy())




class Hertz50Reader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='german_tso_50hertz')
    data_dir = os.path.normpath('GERMAN_TSO/50HERTZ')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('val_type', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('tso', 'VARCHAR'),
               ('pp_id', 'VARCHAR'),
               ('year', 'SMALLINT')]
    tb_pk = ['val_type', 'year', 'hy', 'pp_id']

    exclude_substrings=[]

    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        super().get_fn_list()


    def read(self, fn):

        df_add = pd.read_csv(fn, delimiter=';', index_col=False, skiprows=4)
        df_add = df_add.dropna(how='all', axis=1)

        # get list of value columns
        val_cols = [c for c in df_add.columns
                    if not c in ['Datum', 'Von', 'bis']]

        # convert value columns to float
        str_to_float = lambda x: float(str(x).replace('.', '')
                                             .replace(',', '.'))
        df_add.loc[:, val_cols] = df_add[val_cols].applymap(str_to_float)

        # get dst markers while we have all time columns
        extract_AB = lambda x: ('B' if 'B' in ''.join(x)
                                else 'A' if 'A' in ''.join(x) else '-')
        df_add['dst_hour'] = df_add[['Von', 'bis']].apply(extract_AB, axis=1)

        # get march daylight saving date
        dst_date = df_add.loc[-(df_add.dst_hour == ('-'))].iloc[0]['Datum']


        row_first_AB = df_add.loc[df_add.Datum.isin([dst_date]) & df_add.Von.str.contains('01:')].iloc[-1].name
        row_last_AB = df_add.loc[df_add.Datum.isin([dst_date]) & df_add.Von.str.contains('03:')].iloc[0].name

        df_add.loc[df_add.index.get_values() <= row_first_AB, 'dst_hour'] = '-'
        df_add.loc[df_add.index.get_values() >= row_last_AB, 'dst_hour'] = 'X'
        df_add.loc[:, ['Von', 'bis']] = df_add.loc[:, ['Von', 'bis']].applymap(lambda x: x.replace('-A', '').replace('-B', ''))

        df_add.loc[df_add.Datum.isin([dst_date])]


        # get new df with complete dst switching hours
        dict_dst_new = {'dst_hour': ['A'] * 4 + ['B'] * 4,
                        'Von': ['02:00', '02:15', '02:30', '02:45'] * 2,
                        'bis': ['02:15', '02:30', '02:45', '03:00'] * 2}
        df_add_dst_new = pd.DataFrame.from_dict(dict_dst_new)

        # join all original data to new dataframe
        df_add_dst_new = df_add_dst_new.join(df_add.loc[df_add.Datum == dst_date].set_index(['Von', 'bis', 'dst_hour']), on=['Von', 'bis', 'dst_hour'])
        df_add_dst_new['Datum'] = dst_date

        # add to original dataframe
        df_add = pd.concat([df_add.loc[:row_first_AB],
                            df_add_dst_new,
                            df_add.loc[row_last_AB:]])
        df_add = df_add.reset_index(drop=True)

        # interpolate
        df_add.loc[:, val_cols] = (df_add[val_cols].astype(float)
                                                   .interpolate('cubic',
                                                                axis=0))

        df_add.loc[df_add.Datum.isin([dst_date])].set_index('Von')['MW'].plot()


        # rename columns and drop the obsolete ones
        dict_cols = {'MW': 'DE_WIN_TOT', 'Onshore MW': 'DE_WIN_ONS',
                     'Offshore MW': 'DE_WIN_OFF',
                     'Datum': 'Date', 'Von': 'Time'}
        if 'Solar' in fn:
            dict_cols.update({'MW': 'DE_SOL_PHO'})
        dict_cols = {kk: vv for kk, vv in dict_cols.items()
                     if kk in df_add.columns}
        df_add = df_add[list(dict_cols.keys())].rename(columns=dict_cols)

        # update the value cols list
        val_cols = [dict_cols[c] for c in val_cols]

        # generate datetime column
        dtt_cols = ['Date', 'Time']
        df_add['DateTime'] = df_add[dtt_cols].apply(lambda x: ' '.join(x),
                                                    axis=1)
        # no format: 3.46 s ± 104 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # w/ format: 98 ms ± 491 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        df_add['DateTime'] = pd.to_datetime(df_add.DateTime,
                                            format="%d.%m.%Y %H:%M")
        # drop obsolete cols
        df_add = df_add.drop(dtt_cols, axis=1)

        # convert to UTC
        df_add['DateTime'] = self.tz_localize_convert(df=df_add)

        # average to 1 hour
        df_add.set_index('DateTime').astype(float).resample('H').mean()

        # stack value columns
        df_add = df_add.set_index([c for c in df_add.columns
                                   if not c in val_cols])
        df_add.columns = df_add.columns.rename('pp_id')
        df_add = df_add.stack().rename('value').reset_index()

        # add additional columns
        dict_tp = {'Hochrechnung': 'actual', 'Prognose': 'forecast'}
        df_add['val_type'] = dict_tp[fn.split(os.path.sep)[-1].split('_')[1]]
        df_add['tso'] = '50hertz'

        return df_add


    def postprocessing_tot(self):
        '''
        Various operations once the table df_tot has been assembled.
        '''
        self.df_tot = self.get_hour_of_the_year(self.df_tot)
        self.append_to_sql(self.df_tot.copy())



dict_sql = dict(db='storage2')

kw_dict = dict(dict_sql=dict_sql,
               exclude_substrings=[],
               tm_filt={'year': range(2005, 2018)},
               ext=['csv'])

# %%

if __name__ == '__main__':

    op = Hertz50Reader(kw_dict)
    self = op
    fn = self.fn_list[0]
    self.read_all(skip_sql=True)
    self.postprocessing_tot()
    fn = self._fn
    sys.exit()

# %%

    op = AmprionReader(kw_dict)
    self = op
    fn = self.fn_list[0]
    self.read_all(skip_sql=True)
    self.postprocessing_tot()
    sys.exit()

# %%

    op = TennetReader(kw_dict)
    self = op
    fn = self.fn_list[40]
    self.read_all(skip_sql=True)
    self.postprocessing_tot()
    sys.exit()



# %%

