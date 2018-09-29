#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import pandas as pd
from importlib import reload
from bs4 import BeautifulSoup
import urllib.request
from tqdm import tqdm
import numpy as np

import grimsel_h.auxiliary.timemap as timemap
import grimsel_h.auxiliary.aux_sql_func as aql

import PROFILE_READER.profile_reader as profile_reader

reload(profile_reader)


map_entities = [
 ('CEPS BZ', False), #Czech
 ('NO2 BZ', False),
 ('NO4 BZ', False),
 ('CEPS CA',False), # Czech
 ('CGES BZ', False), #Montenegro
 ('IT-North BZ', 'IT_BZ'),
 ('PSE SA CA', False), 
 ('Ukraine BEI CA', False),
 ('IT-North-FR BZ', 'IT_BZ'),
 ('REE BZ', False), # Spain
 ('Ukraine BEI BZ',  False),
 ('DE-AT-LU',  False), # Aggregate
 ('EMS CA',  False), # Serbia
 ('Elia BZ',  False), # Belgium
 ('50Hertz CA', 'DE_CA'),
 ('DK2 BZ', False),
 ('ESO BZ', False), #Bulgaria
 ('Fingrid BZ', False),
 ('Litgrid BZ', False),
 ('SE2 BZ', False),
 ('ELES BZ', False), # Slovenia
 ('PSE SA BZ', False), # Poland
 ('MEPSO CA', False), # Macedonia
 ('TenneT GER CA', 'DE_CA'),
 ('Elering BZ', False), #Estonia
 ('SE1 BZ',False),
 ('NO3 BZ',False),
 ('Litgrid CA', False),
 ('MAVIR BZ',False), #Hungary
 ('TenneT NL BZ', False), #Netherlands
 ('APG CA', 'AT_CA'),
 ('IPTO BZ', False), #Greece
 ('NOS BiH CA', False), #Bosnia
 ('AST BZ', False), #Latvia
 ('Germany',  'DE_AGG'), #Aggregate
 ('Transelectrica BZ', False), #Romania
 ('Denmark', False),
 ('SE4 BZ', False),
 ('CGES CA', False), #Montenegro
 ('SE3 BZ', False),
 ('EMS BZ', False), #Serbia
 ('OST BZ', False), #Albania
 ('swissgrid BZ', 'CH_BZ'),
 ('SvK CA', False), #Sweden
 ('HOPS CA', False), #Croatia
 ('SEPS BZ', False), #Slovakia
 ('Statnett CA', False),
 ('MAVIR CA', False), #Hungary
 ('NO5 BZ', False),
 ('Energinet CA', False), #Denmarkk
 ('Elia CA', False), #Belgium
 ('ESO CA', False), #Bulgaria
 ('REE CA', False), #Spain
 ('Turkey BZ', False),
 ('MEPSO BZ', False), # Macedonia
 ('Transelectrica CA', False), #Romania
 ('IT-North-CH BZ', 'IT_BZ'),
 ('National Grid BZ', False), #UK
 ('Ireland - (SEM) BZ', False),
 ('IT-North-AT BZ', 'IT_BZ'),
 ('Fingrid CA', False),
 ('IPTO CA', False),
 ('Italy CA', 'IT_AGG'),
 ('Luxembourg',  False),
 ('REN CA', False), #Portugal
 ('HOPS BZ', False), #Croatia
 ('Russia BZ', False),
 ('National Grid CA', False),# UK
 ('Amprion CA', 'DE_CA'),
 ('OST CA', False), #Albania
 ('AST CA', False), #Latvia
 ('IT-GR BZ','IT_BZ'),
 ('Czech Republic', False),
 ('NO1 BZ', False),
 ('RTE BZ', 'FR_BZ'),
 ('DK1  BZ', False),
 ('RTE CA', False),
 ('Netherlands', False),
 ('TransnetBW CA', False),
 ('EirGrid CA', False), #Ireland
 ('TenneT NL CA', False),
 ('NOS BiH BZ', False),
 ('Kaliningrad BZ', False),
 ('Malta BZ', False),
 ('SEPS CA', False), #Slovakia
 ('Elering CA', False), #Estonia
 ('IT-Sicily BZ', 'IT_BZ'),
 ('swissgrid CA', 'CH_CA'),
 ('ELES CA', False), #Slovenia
 ('IT-Brindisi BZ', 'IT_BZ'),
 ('Belarus BZ', False),
 ('France', 'FR_AGG'),
 ('REN BZ', False), #Portugal
 ('Austria', 'AT_AGG'),
 ('CREOS CA', False), #Luxembourg
 ('Poland', False),
 ('Belarus CA', False),
 ('Malta CA', False),
 ('Russia CA', False),
 ('Switzerland', 'CH_AGG')]

df_map_entities = pd.DataFrame(map_entities, columns=['Area0', 'Area'])

class EntsoeCommercialExchangeReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='entsoe_commercial_exchange')
    data_dir = os.path.normpath('ENTSOE/COMMERCIAL_SCHEDULED_EXCHANGE_FTP/')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('nd_to', 'VARCHAR'),
               ('nd_from', 'VARCHAR'),
               ('hy', 'SMALLINT'),
               ('year', 'SMALLINT'),
               ('value', 'DOUBLE PRECISION')]
    tb_pk = ['nd_to', 'nd_from', 'year', 'hy']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read(self, fn):

        df_add = pd.read_excel(fn, header=0, encoding='utf-8')

        df_add = df_add.loc[df_add.AreaInTypeCode.isin(['CTA'])
                          & df_add.AreaOutTypeCode.isin(['CTA'])]

        df_add['nd_to'] = df_add['MapCodeIn'].apply(lambda x: x[:2] + '0')
        df_add['nd_from'] = df_add['MapCodeOut'].apply(lambda x: x[:2] + '0')

        list_nd = ['DE0', 'FR0', 'AT0', 'CH0', 'IT0']

        df_add = df_add.loc[(df_add.nd_to.isin(list_nd)
                             | df_add.nd_from.isin(list_nd))]
                  
        df_add = df_add.pivot_table(index=['year', 'DateTime', 'nd_to', 'nd_from'],
                                   values='Capacity',
                                   aggfunc=sum).reset_index()

        df_add['DateTime'] = pd.to_datetime(df_add['DateTime'])
        df_add['DateTime'] = df_add['DateTime'].dt.tz_localize('UTC')

        df_add['year'] = df_add.DateTime.dt.year

        return df_add

    def get_fn_list(self):
        ''' Remove double files (nd1, nd2) = (nd2, nd1) '''
        
        
        super().get_fn_list()
        
        df_fn = pd.DataFrame(self.fn_list, columns=['fn'])
        df_fn[['nd_1', 'nd_2']] = df_fn.fn.apply(lambda x: tuple(sorted(x.split(os.sep)[-1].split('_')[:2]))).apply(pd.Series)
        df_fn = df_fn.groupby(['nd_1', 'nd_2']).apply(lambda x: x.iloc[0, 0])
        self.fn_list = df_fn.values
       
        
    def post_processing(self):

        self.get_hour_of_the_year(self.df_tot)
        
        self.append_to_sql(self.df_tot.copy())
        

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   exclude_substrings=[],
                   col_filt=[],
                   ext=['xlsx'])

    op = EntsoeCommercialExchangeReader(kw_dict)

    self = op

    fn = self.fn_list[1]

    self.read_all(skip_sql=True)


# %%

class EntsoeCrossBorderReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='entsoe_cross_border')
    data_dir = os.path.normpath('ENTSOE/CROSS_BORDER_PHYSICAL_FLOW/')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('nd_to', 'VARCHAR'),
               ('nd_from', 'VARCHAR'),
               ('hy', 'SMALLINT'),
               ('year', 'SMALLINT'),
               ('value', 'DOUBLE PRECISION')]
    tb_pk = ['nd_to', 'nd_from', 'year', 'hy']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read(self, fn):

        df_add = pd.read_csv(fn).fillna(0)

        df_add['DateTime'] = pd.to_datetime(df_add['Time (UTC)'].apply(lambda x: x.split(' - ')[0]))
        df_add['DateTime'] = df_add['DateTime'].dt.tz_localize('UTC')

        df_add = df_add.drop('Time (UTC)', axis=1)

        df_add = df_add.set_index('DateTime')
        
        new_cols = [tuple(nd.split('(')[-1][:2] + '0' for nd in nd_pair.split(' > ')) for nd_pair in df_add.columns]
        
        df_add.columns = pd.MultiIndex.from_tuples(new_cols,
                                                   names=['nd_from', 'nd_to'])
        df_add = df_add.stack().stack()
        
        df_add = df_add.rename('value')
        
        df_add = df_add.reset_index()

        df_add = df_add.replace({'n/e': np.nan})

#        df_add = df_add.loc[df_add.DateTime == '2015-01-01 00:00:00+00:00']

        return df_add

#    def post_processing(self, df):
#
#        return self.filter_years_by_data_length(df)

    def get_fn_list(self):
        ''' Remove double files (nd1, nd2) = (nd2,nd1) '''
        
        
        super().get_fn_list()
        
        df_fn = pd.DataFrame(self.fn_list, columns=['fn'])
        df_fn[['nd_1', 'nd_2']] = df_fn.fn.apply(lambda x: tuple(sorted(x.split(os.sep)[-1].split('_')[:2]))).apply(pd.Series)
        df_fn = df_fn.groupby(['nd_1', 'nd_2']).apply(lambda x: x.iloc[0, 0])
        self.fn_list = df_fn.values
        

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   exclude_substrings=[],
                   col_filt=[],
                   ext=['csv'])

    op = EntsoeCrossBorderReader(kw_dict)

    self = op

    fn = self.fn_list[0]



    self.read_all()

#    self.df_tot.loc[self.df_tot.nd_id == 'DE0'].pivot_table(values='value', index='hy', columns=['nd_id', 'year']).plot()

# %%

slct_nd = ['CH0', 'AT0']    
    
self.df_tot.loc[self.df_tot.nd_to.isin(slct_nd)
              & self.df_tot.nd_from.isin(slct_nd)
              ]    
    
    
