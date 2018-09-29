#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 09:37:01 2018

@author: user
"""
import os, sys
import numpy as np
import pandas as pd
import itertools
import re
import grimsel.auxiliary.aux_sql_func as aql
import grimsel.auxiliary.maps as maps
import grimsel.auxiliary.timemap as timemap
from xlrd import open_workbook
from grimsel.auxiliary.aux_general import read_xlsx_table

class ProfileReader():

    def __init__(self, **kwargs):

        '''
        Instantiate and manage basic properties of the ProfileReader class.

        Input arguments:
        base_dir -- the base data directory
        exclude_substrings -- list, ignores file names containing any of the
            substrings in this list
        ext -- string, e.g. 'csv'. file extension
        to_sql -- boolean; if True, appends to sql table for each read file
        '''

        base_dir = kwargs.pop('base_dir')


        defaults = dict(ext='csv',
                        exclude_substrings=[],
                        tm_filt={},
                        dict_sql={},
                        col_filt=tuple(),
                        skip_table_init=False)
        for key, val in defaults.items():
            setattr(self, key, val)
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
                kwargs.pop(key)

        self._dir = os.path.join(base_dir, self.data_dir)

        # update sql dict using child defaults
        self.dict_sql.update({kk: vv for kk, vv
                              in self.dict_sql_default.items()
                              if not kk in self.dict_sql.keys()})

        print('self.dict_sql', self.dict_sql)
        print('self.__dict__', self.__dict__.keys())
        print('type(self)', type(self))

        self.init_schema()

        # init table is defined in the child classes since this depends on
        # on the table structure; alternatively, define appropriate
        # child attributes in analogy to dict_sql
        if not self.skip_table_init:
            self.init_table()

    def init_tm(self, yr_list):

        self.tm = timemap.TimeMap(keep_datetime=True)
        self.tm.gen_hoy_timemap(start='{}-1-1 00:00'.format(min(yr_list)),
                                stop='{}-12-31 23:59'.format(max(yr_list)))


    def get_fn_list(self):

        self.fn_list = [os.path.join(self._dir, f)
                        for f in os.listdir(self._dir) if
                        os.path.isfile(os.path.join(self._dir, f))]

        # exclude files whose filenames include excluded substrings
        self.fn_list = [fn for fn in self.fn_list
                        if not any(ss in fn for ss
                                   in self.exclude_substrings)]

        _ext = self.ext if type(self.ext) is list else [self.ext]

        # exclude files with wrong file extension
        self.fn_list = [fn for fn in self.fn_list
                        if fn.split('.')[-1] in _ext]


    def read(self):
        '''
        Main data reading method defined in the children.

        Returns a DataFrame with the required columns
        (DateTime, value) as well as all columns referred to in
        self.col_filt and (probably) defined in the init_table (not stringent)
        '''
        pass

    def time_resample(self, df, res='H'):

        list_ind = [c for c in df.columns if not c in ['DateTime', 'value']]

        df = (df.groupby(list_ind)[['DateTime', 'value']]
                .apply(lambda x: (x.set_index('DateTime').astype(float)
                                  .resample(res).mean())).reset_index())

        return df



    def get_hour_of_the_year(self, df):
        '''
        df must not contain any columns containing time indices < year
        '''

        if not 'year' in df.columns:
            df['year'] = df.DateTime.dt.year

        list_ind = [c for c in df.columns if not ('DateTime' in c or
                                                  'value' in c or 'hy' in c)]



        tm = timemap.TimeMap(keep_datetime=True)
        tm.gen_hoy_timemap(start=df.DateTime.min(),
                           stop=df.DateTime.max())

        # explicitly reset UTC
        tm.df_time_map['DateTime'] = (tm.df_time_map.DateTime
                                        .dt.tz_convert('UTC'))

        # explicitly make datetime
        df['DateTime'] = pd.to_datetime(df.DateTime)

        df = pd.merge(tm.df_time_map[['DateTime', 'hy']],
                      df, on=['DateTime'])

#
#        df.pivot_table(index=)
#
#
#        df = df.set_index(list_ind)
#
#        double_reset = lambda x: (x.reset_index(drop=True)
#                                   .reset_index()['index'])
#
#        def get_hy(x):
#
#            x = x.sort_values('DateTime')
#            x = x.reset_index(drop=True)
#            x = x.reset_index()['index']
#            return x
#
#
#        d
#
#        dfg = df.groupby(level=df.index.names)[['DateTime', 'value']]
#
#        x = dfg.get_group(('transnetbw', 'DE_WIN_TOT', 'actual', 2014))
#
#
#        df['hy'] = dfg.apply(get_hy)

#        df = df.reset_index()

        return df

#df.loc[(df.tso == 'transnetbw')
#              & (df.DateTime.dt.date == datetime.date(2014,4,4))
#              & (df.pp_id == 'DE_WIN_TOT')
#              ]
#
#df.pivot_table(index=['year', 'val_type'], columns=['pp_id'], values='hy', aggfunc=np.max)


    def single_post_processing(self, df_add):
        df_add = self.filter_tm(df_add)
        df_add = self.filter_col(df_add)
        df_add = self.filter_leap_day(df_add)

        return df_add

    def read_all(self, skip_sql=False):

        self.df_tot = pd.DataFrame()

        fn = self.fn_list[0]
        for fn in self.fn_list:
            print('Reading {}'.format(fn))

            self._fn = fn

            df_add = self.read(fn)

            if df_add is None:
                # This happens for the Terna data if one of the automatically
                # downloaded pdf files is not a valid data file.
                continue # nothing to do here



            df_add = self.single_post_processing(df_add)


            if not skip_sql:
                self.df_add = df_add.copy()

                df_add = self.get_hour_of_the_year(df_add)

                df_add = self.post_processing(df_add)

                self.append_to_sql(df_add.copy())

            self.df_tot = pd.concat([self.df_tot, df_add])

    def append_to_sql(self, df):

        if 'DateTime' in df.columns:
            df['DateTime'] = df.DateTime.dt.strftime("%Y-%m-%d %H:%M:%S%z")

        aql.write_sql(df, **self.dict_sql, if_exists='append',
                      chunksize=10000)

    def post_processing(self, df):
        '''
        Operations on the otherwise completed table to avoid clashes
        (e.g. month column addition clashing with hour of the year generation)
        '''
        return df


    def filter_tm(self, df):

        for kk, vv in self.tm_filt.items():
            df = df.loc[getattr(df.DateTime.dt, 'year').isin(vv)]

        return df

    def filter_col(self, df):

        for col, val, sgn in self.col_filt:
            df = df.loc[sgn == df[col].isin(val)]

        return df

    def init_table(self):
        '''
        Initialize output table based on attributes of child classes:
        self.tb_cols and self.tb_pk
        '''

        tb_name = self.dict_sql['tb']
        aql.init_table(tb_name=tb_name, cols=self.tb_cols,
                       schema=self.dict_sql['sc'],
                       db=self.dict_sql['db'],
                       pk=self.tb_pk)

    def init_schema(self):

        exec_str = '''
                   CREATE SCHEMA IF NOT EXISTS {sc};
                   '''.format(**self.dict_sql)
        aql.exec_sql(exec_str, db=self.dict_sql['db'])

    def filter_leap_day(self, df):

        mask_ly = (df.DateTime.dt.day == 29) & (df.DateTime.dt.month == 2)
        df = df[-mask_ly]
        df = df.reset_index(drop=True)

        return df


    def tz_localize_convert(self, tz='Europe/Zurich', df=None):

        _df = self.df_tot if df is None else df

        _df['DateTime'] = _df['DateTime'].dt.tz_localize(tz, ambiguous='infer')
        _df['DateTime'] = _df['DateTime'].dt.tz_convert('UTC')

        if df is not None:
            return _df['DateTime']



    def filter_years_by_data_length(self, df):
        # filter incomplete years
        list_years = (df.loc[df.value > 0].pivot_table(values='value', index='year',
                                     aggfunc=len) < 7000)
        list_years = (list_years.loc[list_years.value]
                                .index
                                .get_level_values(0).values)
        return df.loc[-df.DateTime.dt.year.isin(list_years)]

    def fix_double_dates(self):
        '''
        Terna data has multiple versions of some days.
        '''

        self.df_tot = self.df_tot.sort_values('DateTime')
        self.df_tot = self.df_tot.pivot_table(values='value',
                                              index=['DateTime', 'nd_id',
                                                     'year'],
                                              aggfunc=np.mean)
        self.df_tot = self.df_tot.reset_index()


    def fix_missing_dst(self):
        '''
        Some countries (RTE/Terna) report copied 2am values as 3am values on the last
        Sunday of March, and skip the second 2am values on the last Sunday in
        October.
        '''

        self.df_tot['hour'] = self.df_tot.DateTime.dt.hour
        self.df_tot['doy'] = self.df_tot.DateTime.dt.dayofyear
        self.df_tot['year'] = self.df_tot.DateTime.dt.year

        # remove double march dst hours
        dict_dst = self.tm.get_dst_days(['MAR'])
        list_dst = [(kk[0], vv) for kk, vv in dict_dst.items()]
        mask_double_march = (self.df_tot[['year', 'doy']].apply(tuple, axis=1)
                                                    .isin(list_dst))
        mask_double_march &= self.df_tot.hour.isin([2])
        self.df_tot = self.df_tot.loc[-mask_double_march]

        # double missing dst hours in october
        dict_dst = self.tm.get_dst_days(['OCT'])
        list_dst = [(kk[0], vv) for kk, vv in dict_dst.items()]
        mask_missing_october = (self.df_tot[['year', 'doy']].apply(tuple, axis=1)
                                                       .isin(list_dst))
        mask_missing_october_0 = (mask_missing_october
                                    & self.df_tot.hour.isin([2])
                                    & self.df_tot.DateTime.dt.minute.isin([0]))
        mask_missing_october_1 = (mask_missing_october
                                    & self.df_tot.hour.isin([2])
                                    & self.df_tot.DateTime.dt.minute.isin([30]))
        mask_missing_october = mask_missing_october_0 | mask_missing_october_1

        self.df_tot_oct = pd.concat([self.df_tot.loc[mask_missing_october_0]]
                               + [self.df_tot.loc[mask_missing_october_1]]
                               + [self.df_tot.loc[mask_missing_october_0]]
                               + [self.df_tot.loc[mask_missing_october_1]])
        self.df_tot_oct = self.df_tot_oct.reset_index(drop=True).reset_index()
        self.df_tot_oct.loc[self.df_tot_oct.index.isin([1,2]), 'value'] = np.nan

        self.df_tot = self.df_tot.loc[-(mask_missing_october)]
        self.df_tot = pd.concat([self.df_tot_oct.drop('index', axis=1),
                                 self.df_tot], axis=0)
        self.df_tot = self.df_tot.sort_values(['year', 'doy', 'hour']).reset_index(drop=True)
        self.df_tot['value'] = self.df_tot['value'].interpolate('linear')

        self.df_tot = self.df_tot.reset_index(drop=True)

#        self.df_tot.loc[self.df_tot.doy.isin(dict_dst.values())][['value']].plot(marker='.')



# %%

class EpexPriceVolume(ProfileReader):
    '''
    Reads EPEX price profiles and volumes from csv files.

    It would be better to include the EPEX web parser here.
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='epex_price_volumes')
    data_dir = os.path.normpath('EPEX/MORE')


    # sql table columns
    tb_cols = [('"DateTime"', 'TIMESTAMP'),
                ('nd_id', 'VARCHAR'),
                ('quantity', 'VARCHAR'),
                ('value', 'DOUBLE PRECISION'),
                ('year', 'SMALLINT'),
                ('mt_id', 'SMALLINT'),
#                ('day', 'SMALLINT'),
#                ('hod', 'SMALLINT'),
                ('hy', 'SMALLINT')
                ]

    # sql table primary key
    tb_pk = ['quantity', 'nd_id', 'year', 'hy']

    def __init__(self, kw_dict):

        super().__init__(**kw_dict)

    def post_processing(self, df):

        df['mt_id'] = df['DateTime'].dt.month - 1

        return df



    def read(self, fn):

        df_add = pd.read_csv(fn)

        df_add.columns = [c.replace('/', '_').lower() for c in df_add.columns]

        df_add = df_add.loc[-df_add.year.isin(['year'])]

        # data type conversion
        int_cols = ['year', 'month', 'day', 'hod', 'hoy']
        float_cols = ['price_eur_mwh', 'volume_mwh']
        df_add[int_cols] = df_add[int_cols].applymap(int)
        df_add[float_cols] = df_add[float_cols].applymap(float)

        #
        df_add = df_add.set_index(['region', 'year', 'month', 'day', 'hod', 'hoy']).stack()
        df_add = df_add.reset_index()
        df_add = df_add.rename(columns={'region': 'nd_id', 'level_6': 'quantity', 0: 'value'})

        df_add['nd_id'] = df_add['nd_id'] + '0'
        df_add['DateTime'] = pd.to_datetime(df_add[['year', 'month', 'day', 'hod']].rename(columns={'hod': 'hour'}))

        df_add['DateTime'] = df_add.DateTime.dt.tz_localize('UTC')

        return df_add[['DateTime', 'nd_id', 'year', 'quantity', 'value']]

if __name__ == '__main__':


    kw_dict = dict(dict_sql=dict(db='storage2'),
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[], ext='csv', exclude_substrings=[])

    op = EpexPriceVolume(kw_dict)

    op.get_fn_list()

    self = op

    fn = op.fn_list[1]
    op.read_all()


    sys.exit()


# %%



class RTEProduction(ProfileReader):

    dict_sql_default = dict(sc='profiles_raw', tb='rte_production_eco2mix')
    data_dir = os.path.normpath('SUPPLY_PROFILES/FRANCE_RTE')

    # sql table columns
    tb_cols = [('"DateTime"', 'TIMESTAMP'), ('fl_id', 'VARCHAR'),
                ('value', 'DOUBLE PRECISION'), ('hy', 'SMALLINT'),
                ('year', 'SMALLINT'),
                ('mt_id', 'SMALLINT'),
                ('nd_id', 'VARCHAR(7)')]

    # sql table primary key
    tb_pk = ['nd_id', 'fl_id', 'year', 'hy']

    def __init__(self, kw_dict):

        super().__init__(**kw_dict)


    def post_processing(self):
        
        self.df_tot['mt_id'] = self.df_tot['DateTime'].dt.month - 1

        self.init_tm(self.tm_filt['year'])

        self.fix_missing_dst()

      

        self.df_tot['DateTime'] = self.df_tot.groupby('fl_id')['DateTime'].apply(lambda x: x.dt.tz_localize('Europe/Zurich', ambiguous ='infer'))

        self.df_tot['DateTime'] = self.df_tot.DateTime.dt.tz_convert('UTC')

        
        self.df_tot = self.get_hour_of_the_year(self.df_tot)
        
        self.df_tot = self.filter_tm(self.df_tot)
        
#        
#self.df_tot.loc[(self.df_tot.fl_id == 'photovoltaics')
#                & (self.df_tot.DateTime.dt.month == 3)
#                & (self.df_tot.DateTime.dt.year == 2014)
#                & (self.df_tot.DateTime.dt.day == 30)]
        
        self.df_tot = self.df_tot[['DateTime', 'fl_id', 'value', 'hy',
                                   'year', 'mt_id', 'nd_id']]
        
        self.append_to_sql(self.df_tot)
#     
#self.df_tot.loc[(self.df_tot.year == 2014)
#               & (self.df_tot.fl_id == 'bio_all')
#               & (self.df_tot.hy == 0)
#               ]
#        
#        self.df_tot = self.filter_tm(self.df_tot)

    def single_post_processing(self, df_add):
        df_add = self.filter_col(df_add)
        df_add = self.filter_leap_day(df_add)

        return df_add


    def read(self, fn):

        df_add = pd.read_excel(fn, header=0, encoding='utf-8')

        df_add = df_add.loc[(-df_add.Date.isnull())
                            & (-df_add.Consommation.isnull())]

        df_add['DateTime'] = pd.to_datetime(df_add['Date'].dt.date.apply(str)
                                            + ' ' + df_add['Heures'].apply(str))

        df_add['DateTime'] = df_add['DateTime'].dt.tz_localize('UTC')

        skip_cols = ['DateTime', 'Date', 'Heures', 'Nature', 'Périmètre',
                     'year', 'hour']
        df_add = df_add.set_index(['DateTime'])[[c for c in df_add.columns
                                                 if not c in skip_cols]]


        dict_sf = {'Consommation': 'load',
                   'Prévision J-1': 'load_prediction_d-1',
                   'Prévision J': 'load_prediction_d',
                   'Fioul': 'mineral_oil_heavy',
                   'Charbon': 'hard_coal',
                   'Gaz': 'natural_gas',
                   'Nucléaire': 'nuclear_fuel',
                   'Eolien': 'wind_onshore',
                   'Solaire': 'photovoltaics',
                   'Hydraulique': 'hydro_total',
                   'Pompage': 'pumped_hydro_pumping',
                   'Bioénergies': 'bio_all',
                   'Ech. physiques': 'import_export',
                   'Taux de Co2': 'co2_intensity',
                   'Ech. comm. Angleterre': 'imex_UK',
                   'Ech. comm. Espagne': 'imex_ES',
                   'Ech. comm. Italie': 'imex_IT',
                   'Ech. comm. Suisse': 'imex_CH',
                   'Ech. comm. Allemagne-Belgique': 'imex_DE_BE',
                   'Fioul - TAC': 'mineral_oil_heavy_turbines',
                   'Fioul - Cogén': 'mineral_oil_heavy_chp',
                   'Fioul - Autres': 'mineral_oil_heavy_others',
                   'Gaz - TAC': 'natural_gas_turbines',
                   'Gaz - Cogén.': 'natural_gas_chp',
                   'Gaz - CCG': 'natural_gas_cc',
                   'Gaz - Autres': 'natural_gas_others',
                   'Hydraulique - Fil de l?eau + éclusée': 'run_of_river',
                   'Hydraulique - Lacs': 'reservoirs',
                   'Hydraulique - STEP turbinage': 'pumped_hydro',
                   'Bioénergies - Déchets': 'waste_mix',
                   'Bioénergies - Biomasse': 'biomass',
                   'Bioénergies - Biogaz': 'biogas'}

        col_dict = {'level_1': 'fl_id', 0: 'value'}
        df_add = df_add.stack().reset_index().rename(columns=col_dict)

        df_add = df_add.loc[df_add.fl_id.isin(list(dict_sf.keys()))]

        df_add['fl_id'] = df_add.fl_id.replace(dict_sf)
        df_add['year'] = df_add['DateTime'].dt.year
        df_add['nd_id'] = 'FR0'

        df_add.loc[df_add.value.astype(str) == 'ND'] = np.nan        
        df_add['value'] = df_add['value'].fillna(method='ffill')


        df_add['hour'] = df_add.DateTime.dt.hour
        df_add['date'] = df_add.DateTime.dt.date
        df_add['day'] = df_add.DateTime.dt.day
        df_add['month'] = df_add.DateTime.dt.month
        df_add['year'] = df_add.DateTime.dt.year
        
        df_add = df_add.pivot_table(values='value',
                                    index=['year', 'month', 'day', 'hour',
                                           'fl_id', 'nd_id'], aggfunc=np.mean)
        df_add = df_add.reset_index()
        
        df_add['DateTime'] = pd.to_datetime(df_add[['year', 'month',
                                                    'day', 'hour']])
        

#        df_add['DateTime'] = df_add.DateTime.dt.tz_localize('UTC')        
#df_add.loc[df_add.DateTime.dt.month.isin([10])
#          & df_add.DateTime.dt.day.isin([26])
#          & df_add.fl_id.isin(['bio_all'])
#            ]       

        return df_add

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   tm_filt={'year': [2015, 2014, 2016]},
                   col_filt=[], ext=['xlsx'], exclude_substrings=['Real'])

    pr = RTEProduction(kw_dict)

    pr.get_fn_list()

    pr.fn_list = [f for f in pr.fn_list if '2015' in f or '2016' in f or '2014' in f]

    fn = pr.fn_list[0]
    pr.read_all(skip_sql=True)
    pr.post_processing()



# %%



class RTEProductionRealisProd(ProfileReader):
    '''
    Read production data from RTE RealisationDonneesProduction_*.xls files,
    obtained from http://clients.rte-france.com/lang/fr/visiteurs/vie/prod/realisation_production.jsp
    '''
    
    
    dict_sql_default = dict(sc='profiles_raw', tb='rte_production_realis_prod')
    data_dir = os.path.normpath('SUPPLY_PROFILES/FRANCE_RTE')

    # sql table columns
    tb_cols = [('"DateTime"', 'TIMESTAMP'), ('fl_id', 'VARCHAR'),
                ('value', 'DOUBLE PRECISION'), ('hy', 'SMALLINT'),
                ('year', 'SMALLINT'),
                ('mt_id', 'SMALLINT'),
                ('nd_id', 'VARCHAR(7)')]

    # sql table primary key
    tb_pk = ['nd_id', 'fl_id', 'year', 'hy']

    def __init__(self, kw_dict):

        super().__init__(**kw_dict)

    def post_processing(self, df):

        df['mt_id'] = df['DateTime'].dt.month - 1

        return df

    def read(self, fn):

#
#df_add = pd.read_csv(fn, encoding="ISO-8859-1")
#df_add.apply(lambda x: x.split('\t'), axis=1)
#
##        df_add = df_add.loc[(-df_add.Date.isnull())
##                            & (-df_add.Consommation.isnull())]
#
#        df_add['DateTime'] = pd.to_datetime(df_add['Date'].dt.date.apply(str)
#                                            + ' ' + df_add['Heures'].apply(str))
#
#        df_add['DateTime'] = df_add['DateTime'].dt.tz_localize('UTC')
#
#        skip_cols = ['DateTime', 'Date', 'Heures', 'Nature', 'Périmètre',
#                     'year', 'hour']
#        df_add = df_add.set_index(['DateTime'])[[c for c in df_add.columns
#                                                 if not c in skip_cols]]
#
#        dict_sf = {'Consommation': 'load',
#                   'Prévision J-1': 'load_prediction_d-1',
#                   'Prévision J': 'load_prediction_d',
#                   'Fioul': 'mineral_oil_heavy',
#                   'Charbon': 'hard_coal',
#                   'Gaz': 'natural_gas',
#                   'Nucléaire': 'nuclear_fuel',
#                   'Eolien': 'wind_onshore',
#                   'Solaire': 'photovoltaics',
#                   'Hydraulique': 'hydro_total',
#                   'Pompage': 'pumped_hydro',
#                   'Bioénergies': 'bio_all',
#                   'Ech. physiques': 'import_export',
#                   'Taux de Co2': 'co2_intensity',
##                   'Ech. comm. Angleterre': 'import/export_UK',
##                   'Ech. comm. Espagne': 'import/export_ES',
##                   'Ech. comm. Italie': 'import/export_IT',
##                   'Ech. comm. Suisse': 'import/export_CH',
##                   'Ech. comm. Allemagne-Belgique': 'import/export_DE/BE',
#                   }
#
#
#        col_dict = {'level_1': 'fl_id', 0: 'value'}
#        df_add = df_add.stack().reset_index().rename(columns=col_dict)
#
#        df_add = df_add.loc[df_add.fl_id.isin(list(dict_sf.keys()))]
#
#        df_add['fl_id'] = df_add.fl_id.replace(dict_sf)
#        df_add['year'] = df_add['DateTime'].dt.year
#        df_add['nd_id'] = 'FR0'
#
#        df_add['value'] = df_add.groupby([df_add.fl_id,
#                        df_add.DateTime.dt.hour,
#                        df_add.DateTime.dt.date])['value'].transform(lambda x: x.mean())
#
#        df_add = df_add.loc[df_add.DateTime.dt.minute == 0]


        return df_add

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[], ext=['xls'], exclude_substrings=['eCO2'] + ['%d'%yr for yr in range(2006,2015)])

    pr = RTEProductionRealisProd(kw_dict)

    pr.get_fn_list()

    self = pr

    fn = pr.fn_list[0]
#    pr.read_all()

#    df = aql.read_sql('storage2', 'profiles_raw', 'rte_production',
#                      filt=[('fl_id', ['hard_coal'])])
#
#    df.pivot_table(columns='year', index='mt_id', values='value', aggfunc=np.sum)
#
#    df.set_index('DateTime')['value'].plot()

# %%

class EntsoeGenerationReader(ProfileReader):

    dict_sql_default = dict(sc='profiles_raw', tb='entsoe_generation')
    data_dir = os.path.normpath('ENTSOE/ACTUAL_GENERATION_PER_PRODUCTION_TYPE')

    # sql table columns
    tb_cols = [('"DateTime"', 'TIMESTAMP'), ('fl_id', 'VARCHAR'),
                ('value', 'DOUBLE PRECISION'), ('hy', 'SMALLINT'),
                ('year', 'SMALLINT'), ('nd_id', 'VARCHAR(7)')]

    # sql table primary key
    tb_pk = ['nd_id', 'fl_id', 'year', 'hy']

    def __init__(self, kw_dict):

        super().__init__(**kw_dict)

    def read(self, fn):
        ''' Read data from Entsoe UTC csv files. '''

        df_add = pd.read_csv(fn).fillna(0).replace({'n/e': 0})
        df_add['nd_id'] = df_add['Area'].apply(lambda x:
                                             x.split('(')[-1]
                                              .replace(')', '') + '0')



        # get center datetime column
        separate_dt = lambda x: x.split(' - ')[ind].replace(' (UTC)', '')
        dt_format = '%d.%m.%Y %H:%M'
        for dtcol, ind in [('beg', 0)]:
            df_add[dtcol] = df_add['MTU'].apply(separate_dt)
            df_add[dtcol] = pd.to_datetime(df_add[dtcol], format=dt_format)
            df_add[dtcol] = df_add[dtcol].dt.tz_localize('UTC')

        time_res = 0.5 * pd.Timedelta(df_add['beg'].iloc[1]
                                    - df_add['beg'].iloc[0])
        df_add['DateTime'] = df_add['beg'].map(lambda x: x + time_res)


        # drop unnecessary columns
        df_add = df_add.drop(['MTU', 'Area', 'beg'], axis=1)

        # stack
        df_add = (df_add.set_index(['nd_id', 'DateTime'])
                      .stack()
                      .reset_index()
                      .rename(columns={'level_2': 'fl_id', 0: 'value'}))

        # translate technologies to our sub_fuel names
        dict_sf = {'Biomass': 'bio_all',
                 'Fossil Brown coal/Lignite': 'lignite',
                 'Fossil Coal-derived gas': 'natural_gas',
                 'Fossil Gas': 'natural_gas',
                 'Fossil Hard coal': 'hard_coal',
                 'Fossil Oil': 'mineral_oil_heavy',
                 'Fossil Oil shale': 'mineral_oil_heavy',
                 'Fossil Peat': 'lignite',
                 'Geothermal': 'geothermal',
                 'Pumped': 'pumped_hydro',
                 'Pumped(.+)\.1': 'pumped_hydro_pumping',
                 'Run-of-river': 'run_of_river',
                 'Run-of-river(.+)\.1': 'run_of_river_negative',
                 'Reservoir(.+)\.1': 'reservoir_negative',
                 'Reservoir': 'reservoir',
                 'Marine': 'marine',
                 'Nuclear': 'nuclear_fuel',
                 'Other': 'other',
                 'Other(.+)\.1': 'other_negative',
                 'Other renewable': 'other_ren',
                 'Solar': 'photovoltaics',
                 'Waste': 'waste_mix',
                 'Wind Offshore': 'wind_offshore',
                 'Wind Onshore': 'wind_onshore'}

        # remove sf which are zero
        sf_non_zero = df_add.loc[df_add.value != 0, 'fl_id'].drop_duplicates()
        df_add = df_add.loc[df_add.fl_id.isin(sf_non_zero)]

        # translate sf either based on contains or on regex
        replace_sf = lambda x: x.replace({x.iloc[0]: vv for kk, vv in
                                          dict_sf.items()
                                          if re.match(re.compile(kk),
                                                      x.iloc[0])
                                          or kk in x.iloc[0]})
        df_add['fl_id'] = df_add.groupby('fl_id')['fl_id'].transform(replace_sf)

        #check if any raw entsoe column names remain
        lst_raw = (df_add.loc[df_add.fl_id.str.contains('ggreg'), 'fl_id']
                        .drop_duplicates().tolist())
        if lst_raw:
            raise ValueError('Couldn\'t translate columns {} while reading {}.'
                             .format(', '.join(lst_raw), fn))

        df_add = self.time_resample(df_add, res='H')

        return df_add


if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[])

    nr = EntsoeGenerationReader(kw_dict)
    nr.get_fn_list()

    self = nr

    fn = nr.fn_list[0]
    nr.read_all()




# %%
class OpenPowerSystemReader(ProfileReader):
    ''' Read data from https://data.open-power-system-data.org/. '''

    dict_sql_default = dict(sc='profiles_raw', tb='open_power_system_data')
    data_dir = os.path.normpath('OPEN_POWER_SYSTEM_DATA/TIME_SERIES')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('pp_id', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('val_type', 'VARCHAR'),
               ('tso', 'VARCHAR'),
               ('year', 'SMALLINT')]
    tb_pk = ['pp_id', 'val_type', 'tso', 'year', 'hy']

    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read(self, fn):
        ''' Reads input files and generates a dataframe. '''


        df_add = pd.read_csv(fn)

        lst_tso = ['amprion', 'tennet', 'transnetbw', '50hertz']
        df_add = df_add[['utc_timestamp'] + [c for c in df_add.columns
                         if not 'entsoe' in c
                         and any([tso in c for tso in lst_tso])]]

        df_add['utc_timestamp'] = pd.to_datetime(df_add['utc_timestamp'])
        df_add['DateTime'] = df_add['utc_timestamp'].dt.tz_localize('UTC')
        df_add = df_add.drop('utc_timestamp', axis=1)

        df_add = df_add.set_index('DateTime')


        df_add = df_add.resample('H').mean()

        # split columns names
        col_tuples = [tuple(c.replace('_o', 'o').split('_'))
                      for c in df_add.columns]
        col_level_names = ['nd_id', 'tso', 'pp_id', 'generation', 'val_type']
        df_add.columns = pd.MultiIndex.from_tuples(col_tuples,
                                                   names=col_level_names)
        # stack
        df_add = df_add.stack(level=list(range(5))).rename('value').reset_index()

        df_add = df_add.drop(['generation', 'nd_id'], axis=1)
        df_add.loc[df_add.value < 0] = 0
        df_add = df_add.loc[df_add.pp_id != 0]

        dict_pp_id = {'wind': 'DE_WIN_TOT',
                      'windoffshore': 'DE_WIN_OFF',
                      'windonshore': 'DE_WIN_ONS',
                      'solar': 'DE_SOL_PHO'}

        df_add['pp_id'] = df_add['pp_id'].replace(dict_pp_id)

        return df_add

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[],
                   exclude_substrings=['filtered', '30', '60'])

    op = OpenPowerSystemReader(kw_dict)

    self = op

    fn = self.fn_list[0]

    sys.exit()

    op.read_all(skip_sql=True)
    self.df_tot = self.get_hour_of_the_year(self.df_tot)

    self.append_to_sql(self.df_tot)



# %%
class CHPProfileReader(ProfileReader):
    ''' '''

    dict_sql_default = dict(sc='profiles_raw', tb='chp_profiles')
    data_dir = os.path.normpath('HEAT_DEMAND_PROFILE')

    tb_cols = [('nd_id', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT')]
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
    
        wb = open_workbook(fn)
        profile_heat_cols = range(31)
        df_profile_heat = read_xlsx_table(wb, ['TOT'], columns=profile_heat_cols)
    
        years = df_profile_heat.loc[1]
        ctrys = df_profile_heat.loc[0]
        cols =  [(years[i], ctrys[i]) for i in range(len(years))]
    
        df_profile_heat.columns = ['hy'] + cols[1:]
        df_profile_heat = df_profile_heat.drop([0,1], axis=0).reset_index(drop=True)
        df_profile_heat = df_profile_heat.set_index('hy')
        df_profile_heat.columns = pd.MultiIndex.from_tuples(df_profile_heat.columns,
                                                            names=['year','country'])
    
        df_profile_heat = df_profile_heat.stack(['year', 'country']).reset_index()
    
        df_profile_heat = df_profile_heat.loc[df_profile_heat.year == 2015]
    
        df_profile_heat.columns = ([c for c in df_profile_heat.columns[:-1]]
                                 + ['power'])
        df_profile_heat['value'] = df_profile_heat['power'].apply(float)
            
        df_profile_heat = df_profile_heat.loc[df_profile_heat.year == 2015]
    
        df_profile_heat['nd_id'] = df_profile_heat['country'] + '0'
    
        df_profile_heat = df_profile_heat[['hy', 'nd_id', 'value']]
        
        # normalize  
        df_profile_heat['value'] = df_profile_heat.groupby('nd_id')['value'].apply(lambda x: x/x.sum())
        
        return df_profile_heat

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[],
                   exclude_substrings=[], ext='xlsx')

    op = CHPProfileReader(kw_dict)

    self = op

    fn = self.fn_list[0]

#    sys.exit()

    op.read_all(skip_sql=True)

    self.append_to_sql(self.df_tot)



# %%
class WeeklyRORReader(ProfileReader):
    ''' '''

    dict_sql_default = dict(sc='profiles_raw', tb='weekly_ror_data')
    data_dir = os.path.normpath('WEEKLY_ROR_DATA')

    tb_cols = [('nd_id', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT')]
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
    
        wb = open_workbook(fn)
        weekly_ror_production_cols = ['doy', 'CH0', 'AT0', 'DE0', 'FR0', 'IT0']
        df_daily_ror = read_xlsx_table(wb, ['ALL'], columns=weekly_ror_production_cols)
        df_daily_ror = df_daily_ror.set_index('doy').unstack().reset_index()
        df_daily_ror.columns = ['nd_id', 'doy', 'week_ror_output']
        
        self.init_tm([2015])
        
        df_daily_ror = pd.merge(self.tm.df_time_map[['doy', 'hy']],
                                df_daily_ror, on='doy', how='outer')
        df_daily_ror.columns = ['doy', 'hy', 'nd_id', 'value']
        
        df_daily_ror['value'] = df_daily_ror.groupby('nd_id')['value'].apply(lambda x: x/x.sum())
        
        return df_daily_ror[['nd_id', 'hy', 'value']]

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[],
                   exclude_substrings=[], ext='xlsx')

    op = WeeklyRORReader(kw_dict)

    self = op

    fn = self.fn_list[0]

#    sys.exit()

    op.read_all(skip_sql=True)

    self.append_to_sql(self.df_tot)


# %%
class NinjaReader(ProfileReader):
    ''' Read data from www.renewables.ninja. '''

    dict_sql_default = dict(sc='profiles_raw', tb='ninja')
    data_dir = os.path.normpath('NINJA')

    exclude_substrings = ['near', 'long', 'future']

    tb_cols = [('"DateTime"', 'TIMESTAMP'), ('pt_id', 'VARCHAR'),
                ('pp_id', 'VARCHAR'), ('value', 'DOUBLE PRECISION'),
                ('proftype', 'VARCHAR'), ('hy', 'SMALLINT'),
                ('year', 'SMALLINT'), ('nd_id', 'VARCHAR(7)'),
                ('location', 'VARCHAR')]
    tb_pk = ['nd_id', 'pt_id', 'proftype', 'location', 'year', 'hy']


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()

    def read(self, fn):
        '''
        Reads input files and generates a dataframe with columns
        (datetime, type, value), with type as (onshore, offshore, national)
        '''


        read_kws = dict(skiprows=1, header=1)
        df_add = pd.read_csv(fn, delimiter=',', **read_kws)

        fn_split = [fs.split(os.sep) for fs in fn.split('_')]
        fn_split = list(itertools.chain(*fn_split))
        fn_pp_type = [fc for fc in fn_split
                      if any(ss in fc for ss in
                             ['WIN', 'SOL', 'OFF', 'ONS',
                              'PHO', 'wind', 'solar'])]

        fn_node = [fc for fc in fn_split
                   if any(ss == fc for ss in ['FR', 'DE', 'CH', 'AT', 'IT'])]

        col_dict = {'time': 'DateTime', 'national': 'WIN_TOT',
                    'offshore': 'WIN_OFF', 'onshore': 'WIN_ONS',
                    0: 'value', 'level_1': 'pt_id', 'UTC': 'DateTime'}

        if ('national' in df_add.columns
            and not any([pt in df_add.columns for pt in ['wind', 'solar',
                                                         'onshore', 'offshore']])):
            # no hints about pt from columns -> use file name
            if 'wind' in fn_split:
                col_dict.update({'national': 'WIN_ONS'})
            if 'pv' in fn_split:
                col_dict.update({'national': 'SOL_PHO'})



        if any([ss in fn for ss in ['_pv_', '_wind_']]): # is national solar or wind data
            drop_cols = []
            add_cols = dict(proftype='national', location='national')

        else: # is location
            drop_cols = ['Europe/Paris']
            pp_type = '_'.join(fn_pp_type)
            col_dict.update({'kW': pp_type})
            add_cols = dict(proftype='location',
                            location=fn_split[-1].replace('.' + self.ext, ''))

        add_cols.update({'nd_id': fn_node[0] + '0'})

        df_add = (df_add.rename(columns=col_dict).set_index('DateTime')
                        .drop(drop_cols, axis=1)
                        .stack().reset_index()
                        .rename(columns=col_dict))

        for col_name, col_val in add_cols.items():
            df_add[col_name] = col_val

        # convert DateTime data type to datetime
        df_add['DateTime'] = pd.to_datetime(df_add.DateTime)
        df_add['DateTime'] = df_add['DateTime'].dt.tz_localize('UTC')

        return df_add


    def merge_location_profiles_sql(self):

        exec_str = '''
                   INSERT INTO {sc}.{tb}("DateTime", pt_id, value,
                                         proftype, hy, year, nd_id, location)
                   SELECT "DateTime", pt_id, AVG(value) AS value, proftype,
                          hy, year, nd_id, 'national'::VARCHAR AS location
                   FROM profiles_raw.ninja
                   WHERE proftype = 'location' AND NOT location = 'national'
                   GROUP BY "DateTime", pt_id, proftype, hy, year, nd_id;

                   DELETE FROM {sc}.{tb}
                   WHERE proftype = 'location' AND NOT location = 'national';
                   '''.format(**self.dict_sql)
        aql.exec_sql(exec_str, db=self.dict_sql['db'])

    def make_pp_id_col(self):
        aql.exec_sql('''
                     UPDATE {sc}.{tb}
                     SET pp_id = SUBSTRING(nd_id FROM 1 FOR 2) || '_' || pt_id;
                     '''.format(**self.dict_sql), db=self.dict_sql['db'])

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(exclude_substrings=['near', 'long', 'future'],
                   dict_sql=dict_sql, tm_filt={'year': range(2005, 2018)},
                   col_filt=[('pt_id', ['WIN_TOT'], False)])


    nr = NinjaReader(kw_dict)
    self = nr
    fn= self.fn_list[0]

    nr.read_all()
    nr.make_pp_id_col()
    nr.merge_location_profiles_sql()


    # add 2017

    dict_cap_compl = {
                      ('IT0', 'WIN_ONS', 2017): 9383.933906 + 359.2, # +359.2 MW http://www.qualenergia.it/articoli/20180213-eolico-i-dati-2017-sull-installato-italia-europa-e-nel-mondo-
                      ('IT0', 'SOL_PHO', 2017): 19283.173 + 409, # + 409 http://www.solareb2b.it/nel-2017-italia-installati-409-mw-nuovi-impianti-fv-11/
                      ('CH0', 'SOL_PHO', 2017): 1660.21 + 260, # +260 MW https://www.energie-cluster.ch/admin/data/files/file/file/2090/180112_mm_markt17.pdf?lm=1516007459
                      ('CH0', 'WIN_ONS', 2017): 74.9, # no change http://www.suisse-eole.ch/de/windenergie/statistik/
                      ('FR0', 'SOL_PHO', 2017): 7647, # https://www.rte-france.com/sites/default/files/panorama-31mars18.pdf
                      ('FR0', 'WIN_ONS', 2017): 13539, # https://www.rte-france.com/sites/default/files/panorama-31mars18.pdf
                      ('AT0', 'SOL_PHO', 2017): 1089.529000 + 153, # http://www.iea-pvps.org/fileadmin/dam/public/report/statistics/IEA-PVPS_-_A_Snapshot_of_Global_PV_-_1992-2017.pdf
                     }



    mps = maps.Maps('lp_input_calibration_years', 'storage2')
    cols = ['pp_id'] + [c for c in aql.get_sql_cols('plant_encar', 'lp_input_calibration_years', 'storage2').keys()
                        if 'cap_pwr_leg' in c]
    dfcap = aql.read_sql('storage2', 'lp_input_calibration_years', 'plant_encar')[cols]
    dfcap = dfcap.rename(columns={'cap_pwr_leg': 'cap_pwr_leg_yr2015'})
    dfcap['pp_id'].replace(mps.dict_pp, inplace=True)
    dfcap = dfcap.loc[dfcap.pp_id.str.contains('WIN|SOL')]
    dfcap['nd_id'] = dfcap['pp_id'].apply(lambda x: x[:2] + '0')
    dfcap['pt_id'] = dfcap['pp_id'].apply(lambda x: x[3:])

    # add missing values 2017:
    dfcap = dfcap.set_index(['nd_id', 'pt_id'])
    for kk, vv in dict_cap_compl.items():
        print(kk, vv)
        dfcap.loc[kk[:2], 'cap_pwr_leg_yr2017'] = vv

    dfcap = dfcap.reset_index()
    dfcap = dfcap.drop('pp_id', axis=1).set_index(['nd_id', 'pt_id']).stack().reset_index().rename(columns={'level_2': 'year', 0: 'cap_end'})
    dfcap['year'] = dfcap['year'].apply(lambda x: int(x[-4:]))
    dfcap = dfcap.sort_values(['nd_id', 'pt_id', 'year'])
    dfcap['cap_beg'] = dfcap.groupby(['nd_id', 'pt_id'])['cap_end'].transform(lambda x: x.shift().fillna(method='bfill'))
    dfcap['scale_beg'] = dfcap['cap_beg'] / dfcap['cap_end']
    dfcap['scale_end'] = 1
    dfcap.loc[dfcap.scale_beg == float('inf'), 'scale_beg'] = 1

    aql.write_sql(dfcap, 'storage2', 'public', 'temp_add_column', 'replace')

    exec_str = '''
               ALTER TABLE profiles_raw.ninja
               ADD COLUMN IF NOT EXISTS scale DOUBLE PRECISION,
               ADD COLUMN IF NOT EXISTS value_sc DOUBLE PRECISION;

               UPDATE profiles_raw.ninja AS tbnn
               SET scale = COALESCE((tbsc.scale_end - tbsc.scale_beg) / (8760 - 1) * hy + tbsc.scale_beg, 1)
               FROM temp_add_column AS tbsc
               WHERE tbnn.year = tbsc.year
                   AND tbnn.pt_id = tbsc.pt_id
                   AND tbnn.nd_id = tbsc.nd_id;

               UPDATE profiles_raw.ninja
               SET value_sc = scale * value;
               '''
    aql.exec_sql(exec_str, db=dict_sql['db'])

    aql.read_sql(dict_sql['db'], 'profiles_raw', 'ninja', filt=[('year', [2016])])


    filt=[('pt_id', ['WIN_OFF']), ('nd_id', ['DE0']), ('year', [2015])]
    df = aql.read_sql('storage2', 'profiles_raw', 'ninja', filt=filt)
    dfcap = aql.read_sql('storage2', 'public', 'temp_add_column', filt=filt)

    exec_strg = '''
                SELECT pt_id, proftype, year, nd_id, location, AVG(value_sc), AVG(value)
                FROM profiles_raw.ninja
                GROUP BY pt_id, proftype, year, nd_id, location
                '''
    cfnn = pd.DataFrame(aql.exec_sql(exec_strg, db='storage2'),
                 columns=['pt_id', 'proftype', 'year', 'nd_id', 'location', 'cfsc', 'cf'])

    cols = ['pp_id'] + [c for c in aql.get_sql_cols('plant_encar', 'lp_input_calibration_years', 'storage2').keys()
                        if 'cf_max' in c]
    dfcf = aql.read_sql('storage2', 'lp_input_calibration_years', 'plant_encar')[cols]
    dfcf = dfcf.rename(columns={'cf_max': 'cf_max_yr2015'})
    dfcf['pp_id'].replace(mps.dict_pp, inplace=True)
    dfcf = dfcf.loc[dfcf.pp_id.str.contains('WIN|SOL')]
    dfcf['nd_id'] = dfcf['pp_id'].apply(lambda x: x[:2] + '0')
    dfcf['pt_id'] = dfcf['pp_id'].apply(lambda x: x[3:])
    dfcf = dfcf.drop('pp_id', axis=1).set_index(['nd_id', 'pt_id']).stack().reset_index().rename(columns={'level_2': 'year', 0: 'cfmax'})
    dfcf['year'] = dfcf['year'].apply(lambda x: int(x[-4:]))
    dfcf = dfcf.set_index(['nd_id', 'pt_id', 'year'])

    cfnn = cfnn.join(dfcf, on=dfcf.index.names)
# %%

if __name__ == '__main__':

    # %%
    ax = cfnn.loc[cfnn.pt_id.isin(['SOL_PHO'])
           & cfnn.proftype.isin(['national'])
           & -cfnn.year.isin([2006, 2005])].pivot_table(index=['nd_id', 'year'], columns=['pt_id', 'proftype'], values=['cf', 'cfsc', 'cfmax'], aggfunc=sum).plot(marker='o')
    ax.set_ylim(bottom=0)
    # %%
    filt = [('pt_id', ['WIN_OFF']), ('nd_id', ['DE0']), ('year', [2011])]
    aql.read_sql('storage2', 'profiles_raw', 'ninja', filt=filt).pivot_table(index=['hy'], columns=['pt_id'], values=['value_sc'], aggfunc=sum).plot()




