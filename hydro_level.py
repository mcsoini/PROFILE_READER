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
import urllib.request
from tqdm import tqdm
import numpy as np

import grimsel.auxiliary.timemap as timemap
import grimsel.auxiliary.aux_sql_func as aql

import PROFILE_READER.profile_reader as profile_reader

reload(profile_reader)

class FRHydroLevelReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='hydro_level_rte_fr')
    data_dir = os.path.normpath('HYDRO_FILLING_LEVELS/RTE_FRANCE')

    tb_cols = [('nd_id', 'VARCHAR'),
               ('start_level_interp', 'DOUBLE PRECISION'),
               ('average_level', 'DOUBLE PRECISION'),
               ('start_level_diff', 'DOUBLE PRECISION'),
               ('wk_id', 'SMALLINT'),
               ('year', 'SMALLINT')]
    tb_pk = ['nd_id', 'year', 'wk_id']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read_all(self):
        '''
        Simplified read_all since we don't need hourly data.
        '''
        self.df_tot = pd.DataFrame()
        fn = self.fn_list[0]
        for fn in self.fn_list:
            print('Reading {}'.format(fn))

            self._fn = fn

            df_add = self.read(fn)



            self.df_tot = pd.concat([self.df_tot, df_add])

        self.post_processing()

        self.append_to_sql(self.df_tot.copy())


    def read(self, fn):

        df_add = pd.read_csv(fn, delimiter='\t', skiprows=3, skipfooter=1,
                             engine='python', names=['wk_id', 'average_level'])
       
        with open(fn, 'r', encoding='latin-1') as f:
            year = f.readline().replace('\n', '').split(' ')[-1]
        
        
        df_add['year'] = int(year)
        df_add['nd_id'] = 'FR0'
        df_add['average_level'] = df_add.average_level.astype(float)
        df_add['wk_id'] = df_add.wk_id.astype(int) - 1

        return df_add

    def post_processing(self):
#        
        
        self.df_tot = self.df_tot.sort_values(['year', 'wk_id'])
        self.df_tot = self.df_tot.reset_index(drop=True)
        
        self.df_tot.loc[:, 'start_level_interp'] = (self.df_tot['average_level']
                                                 .rolling(2)
                                                 .mean())
        self.df_tot['start_level_diff'] = self.df_tot['start_level_interp'].diff().shift(-1)
        self.df_tot = self.df_tot.loc[self.df_tot.year.isin(self.tm_filt['year'])]

if __name__ == '__main__':
#    sys.exit()

    kw_dict = dict(dict_sql=dict(db='storage2'),
                   exclude_substrings=[],
                   tm_filt={'year': range(2015, 2018)},
                   ext=['csv'])


    op = FRHydroLevelReader(kw_dict)
    self = op
    fn = self.fn_list[0]
#    op.read_all()
#
#
#ax = self.df_tot.set_index(['year', 'wk_id'])[['average_level']].plot(drawstyle="steps-mid")
#self.df_tot.set_index(['year', 'wk_id'])[['average_level']].plot(marker='x', ax=ax, linewidth=0)
#self.df_tot.set_index(['year', 'wk_id'])[['start_level_interp']].plot(marker='.', ax=ax)
#


# %%

class EntsoeHydroLevelReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='hydro_level_entsoe')
    data_dir = os.path.normpath('HYDRO_FILLING_LEVELS/ENTSOE')

    tb_cols = [('nd_id', 'VARCHAR'),
               ('start_level_interp', 'DOUBLE PRECISION'),
               ('average_level', 'DOUBLE PRECISION'),
               ('start_level_diff', 'DOUBLE PRECISION'),
               ('wk_id', 'SMALLINT'),
               ('year', 'SMALLINT')]
    tb_pk = ['nd_id', 'year', 'wk_id']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read_all(self):
        '''
        Simplified read_all since we don't need hourly data.
        '''
        self.df_tot = pd.DataFrame()
        fn = self.fn_list[0]
        for fn in self.fn_list:
            print('Reading {}'.format(fn))

            self._fn = fn

            df_add = self.read(fn)



            self.df_tot = pd.concat([self.df_tot, df_add])

        self.post_processing()

        self.append_to_sql(self.df_tot.copy())


    def read(self, fn):
#       
        
        fill_nan = {'AT0': {2015: 2016}}



        

        df_add = pd.read_excel(fn, header=7)
        df_add = df_add.loc[~df_add.index.get_level_values(0).isnull()]
                
        nd = fn.replace('.xlsx', '').split('_')[-1].upper() + '0'
        
        df_add = df_add.loc[:, -df_add.isnull().all(axis=0)]
        df_add = df_add.replace({'-': np.nan})
        df_add = df_add.astype(float)

        df_add.columns = map(int, df_add.columns)
        
        
        for col, filling in fill_nan[nd].items() if nd in fill_nan.keys() else {}:
            df_add[col].fillna(df_add[filling], inplace=True)


        df_add = df_add.stack().reset_index()
        
        df_add.columns = ['wk_id', 'year', 'average_level']
        
        df_add['wk_id'] = df_add.wk_id.apply(lambda x: int(x.split(' ')[-1]) - 1)
        
        df_add.pivot_table(index='wk_id', values='average_level', columns='year').plot()
        
        
        df_add['nd_id'] = nd
        
#        with open(fn, 'r') as f:
#            year = f.readline().replace('\n', '').split(' ')[-1]
#        
#        
#        df_add['year'] = int(year)
#        df_add['nd_id'] = 'FR0'
#        df_add['average_level'] = df_add.average_level.astype(float)
#        df_add['wk_id'] = df_add.wk_id.astype(int) - 1

        return df_add

    def post_processing(self):
        
        
        self.df_tot['start_level_interp'] = (self.df_tot.sort_values(['year', 'wk_id'])['average_level']
                                                 .rolling(2)
                                                 .mean())
        self.df_tot['start_level_diff'] = self.df_tot['start_level_interp'].diff().shift(-1)
        self.df_tot = self.df_tot.loc[self.df_tot.year.isin(self.tm_filt['year'])]

if __name__ == '__main__':
#    sys.exit()

    kw_dict = dict(dict_sql=dict(db='storage2'),
                   exclude_substrings=[],
                   tm_filt={'year': range(2015, 2018)},
                   ext=['xlsx'])


    op = EntsoeHydroLevelReader(kw_dict)
    self = op
    fn = self.fn_list[3]
#    op.read_all()

#
#ax = self.df_tot.set_index(['year', 'wk_id'])[['average_level']].plot(drawstyle="steps-mid")
#self.df_tot.set_index(['year', 'wk_id'])[['average_level']].plot(marker='x', ax=ax, linewidth=0)
#self.df_tot.set_index(['year', 'wk_id'])[['start_level_interp']].plot(marker='.', ax=ax)


    df = pd.DataFrame(aql.exec_sql('''
                              SELECT * FROM profiles_raw.entsoe_generation
                              WHERE nd_id = 'AT0' AND fl_id = 'reservoir' AND year = 2015
                              ''', db='storage2'), columns=['datetime', 'fl_id', 'value', 'hy', 'year', 'nd_id'])


