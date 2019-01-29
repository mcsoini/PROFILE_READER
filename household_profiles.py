# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:16:01 2019

@author: mcsoini
"""


import os, sys
import numpy as np
import pandas as pd
import itertools
import re
import grimsel.auxiliary.sqlutils.aux_sql_func as aql
import grimsel.auxiliary.maps as maps
import grimsel.auxiliary.timemap as timemap
from xlrd import open_workbook
from grimsel.auxiliary.aux_general import read_xlsx_table
import hashlib
import datetime

from PROFILE_READER.profile_reader import ProfileReader
import PROFILE_READER.config as conf


base_dir = conf.BASE_DIR



# %%
class CREMProfileReader(ProfileReader):
    ''' '''

    dict_sql_default = dict(sc='profiles_raw', tb='households_crem')
    data_dir = os.path.normpath('HOUSEHOLD_PROFILES_CREM\\RAW_DATA')

    tb_cols = [('nd_id', 'VARCHAR'),
               ('erg_tot', 'DOUBLE PRECISION'),
               ('erg_tot_filled', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('missing_incl_tz', 'FLOAT')]
    tb_pk = ['nd_id', 'hy']

    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def single_post_processing(self, df):
        '''
        Overwriting parent method.
        '''
        return df
        

    def read(self, fn):
        ''' Reads input files and generates a dataframe. '''
    
        df = pd.read_excel(os.path.join(fn), header=1, sheet_name='Feuil1')
        df.columns = ['obj', 'place', 'date', 'time', 'erg_tot', 'erg1', 'erg2']
        df = df.loc[~(df.time.isin(['Résultat']) | df.obj.isin(['Résultat global']))]
        
        obj, place = df.iloc[0][['obj', 'place']].values
        
        df[['date']] = df[['date']].fillna(method='ffill', axis=0)
        
        # make CET non-localized DateTime column
        dt_kwargs = dict(freq='15min', start='2015-01-01 00:00:00', end='2015-12-31 23:59:59', tz='CET')
        dftm = pd.DataFrame(index=pd.DatetimeIndex(**dt_kwargs).rename('DateTime')).reset_index()
        
        # convert DateTime in input table to actual DateTime
        df['DateTime'] = pd.to_datetime(df.date + ' ' + df.time, format='%d.%m.%Y %H:%M:%S')
        
        df.drop(['date', 'time', 'erg1', 'erg2', 'obj', 'place'], axis=1, inplace=True)
        
        # round DateTime to 15 minutes
        df['DateTime'] = df['DateTime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour, 15*(dt.minute // 15)))

        # add nans for missing dst switching hours, if necessary
        oct_hours = df.loc[df.DateTime.dt.month.isin([10])
                         & df.DateTime.dt.day.isin([25])
                         & df.DateTime.dt.hour.isin([2])]
        
        if len(oct_hours) == 4:
            
            df = pd.concat([df.loc[:oct_hours.index.min() - 1],
                            oct_hours,
                            oct_hours.assign(erg_tot=np.nan),
                            df.loc[oct_hours.index.max() + 1:]], axis=0)


        # localize
        df['DateTime'] = df.DateTime.dt.tz_localize('CET', ambiguous='infer')
        
        # join to time map -> expanded table with NaNs
        df = pd.merge(dftm, df, on='DateTime', how='outer')
        
        # 
        df['DateTime'] = df.DateTime.dt.tz_convert('UTC')
        
        df = df.loc[df.DateTime.dt.year == 2015]
        
        # make UTC DateTime column so we fill the slots missing due to tz conversion
        dt_kwargs = dict(freq='15min', start='2015-01-01 00:00:00', end='2015-12-31 23:59:59', tz='UTC')
        dftm = pd.DataFrame(index=pd.DatetimeIndex(**dt_kwargs).rename('DateTime')).reset_index()
        
        # join to time map -> expanded table with NaNs for UTC
        df = pd.merge(dftm, df, on='DateTime', how='outer')
        
        
        df['week'] = df.DateTime.dt.week
        df['how'] = df.DateTime.dt.dayofweek * 24 + df.DateTime.dt.hour + df.DateTime.dt.minute / 60
        
        df['erg_tot'] = pd.to_numeric(df.erg_tot, errors='coerce', downcast='float')
        
        sum_keep_nan = lambda x: np.nan if np.nan in x else sum(x)
        dfpv = df.pivot_table(index='how', columns='week', values='erg_tot', aggfunc=sum_keep_nan)
        dfpv = dfpv.fillna(method='ffill', axis=1).stack().rename('erg_tot_filled')
        
        df = df.join(dfpv, on=['how', 'week'])

        df['nd_id'] = hashlib.md5(str((obj, place)).encode('utf-8')).hexdigest()[:10].upper()
        df['missing_incl_tz'] = sum(df.erg_tot.isna()) / len(df.erg_tot)
    
    
        
        return df

if __name__ == '__main__':

    dict_sql = dict(db='storage2')



    kw_dict = dict(dict_sql=dict_sql, base_dir=base_dir,
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[],
                   exclude_substrings=[], ext='xlsx')

    op = CREMProfileReader(kw_dict)

    self = op

    fn = self.fn_list[0]

#    sys.exit()

#    op.read_all()

#    self.append_to_sql(self.df_tot)



# %%
    

class SwissLocationSolarReader(ProfileReader):
    ''' '''

    dict_sql_default = dict(sc='profiles_raw', tb='swiss_location_solar')
    data_dir = os.path.normpath('PV_CANTONS_SYNTHETIC')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('year', 'SMALLINT'),
               ('canton', 'VARCHAR'),
               ]
    tb_pk = ['canton', 'hy', 'year']

    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def single_post_processing(self, df):
        '''
        Overwriting parent method.
        '''
        return df
        

    def get_fn_list(self):
        
        gen = os.walk(self._dir)
        gen.__next__()
        
        self.fn_list = list(os.sep.join((p[0], p[2][0])) for p in gen)

    def read_all(self):

        df = pd.concat([pd.read_csv(fn, sep=';').assign(canton=fn.replace('.csv', '').split('_')[-1]) for fn in self.fn_list])    

        dfpv = df.pivot_table(index='Date', values='Generation', columns='canton')
        dfpv.index = pd.to_datetime(dfpv.index)
        dfpv = dfpv.resample('h').mean()
        dfpv = dfpv.loc[~dfpv.index.isin(dfpv.loc['2016-02-29'].index.get_level_values('Date'))]
        
        dfpv.index = dfpv.index.tz_localize('UTC')

        df = dfpv.stack().rename('value').reset_index()
        df = df.rename(columns={'Date': 'DateTime'})

        df = self.get_hour_of_the_year(df)
   
        self.append_to_sql(df)
     

if __name__ == '__main__':

    dict_sql = dict(db='storage2')



    kw_dict = dict(dict_sql=dict_sql, base_dir=base_dir,
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[],
                   exclude_substrings=[], ext='xlsx')

    op = SwissLocationSolarReader(kw_dict)

    self = op

    fn = self.fn_list[0]

#    sys.exit()

    op.read_all()

#    self.append_to_sql(self.df_tot)


