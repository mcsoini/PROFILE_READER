#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:48:40 2018

@author: user
"""

import profile_reader

class SwissGrid15MinReader(profile_reader.ProfileReader):
    '''
    Read load data from Swissgrid EnergieUebersicht*.xls files,
    obtained from https://www.swissgrid.ch/de/home/operation/grid-data.html
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='swissgrid_load')
    data_dir = os.path.normpath('DEMAND/SWITZERLAND')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('nd_id', 'VARCHAR'), ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'), ('year', 'SMALLINT')]
    tb_pk = ['nd_id', 'year', 'hy']

    exclude_substrings=['final']


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()

    def read(self, fn):
        ''' Reads input files and generates a dataframe. '''

        df_add = pd.read_excel(fn, sheet_name='Zeitreihen1h00', usecols='A:B', header=1).dropna()
        df_add.columns = ['DateTime', 'value']

        df_add['DateTime'] = pd.to_datetime(df_add['DateTime'])

        # Swissgrid repeats the hour 3 instead of the hour 2 for the
        # switch dst -> normal time; this needs to be fixed in order for
        # tz_localize to work
        dst_index = df_add.loc[df_add.DateTime.dt.month.isin([10]) # october
                             & df_add.DateTime.dt.dayofweek.isin([6]) # sunday
                             & df_add.DateTime.dt.hour.isin([3]) # faulty switch hour
                             ].iloc[-2].name
        df_add.loc[dst_index, 'DateTime'] -= pd.Timedelta(1, 'h')

        df_add['DateTime'] = df_add['DateTime'].dt.tz_localize('Europe/Zurich', ambiguous='infer')
        df_add['DateTime'] = df_add['DateTime'].dt.tz_convert('UTC')

        df_add['nd_id'] = 'CH0'

        return df_add

    def postprocessing_tot(self):

        self.df_tot['year'] = self.df_tot.DateTime.dt.year
        self.df_tot = self.filter_years_by_data_length(self.df_tot)
        self.df_tot = self.get_hour_of_the_year(self.df_tot)
        self.append_to_sql(self.df_tot)


if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   col_filt=[], ext='xls')

    op = SwissGridLoadReader(kw_dict)
    self = op
    op.read_all(skip_sql=True)
    op.postprocessing_tot()


