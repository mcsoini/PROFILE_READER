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
import PyPDF2
import numpy as np

import grimsel.auxiliary.timemap as timemap
import grimsel.auxiliary.aux_sql_func as aql

import PROFILE_READER.profile_reader as profile_reader

reload(profile_reader)

# %% Complement days

class ProfileYearAdder(profile_reader.ProfileReader):
    '''
    Adds missing years to profile tables and complements gaps in the existing
    years' data.
    Makes use of some ProfileReader methods, but doesn't call its __init__.
    '''

    all_yr = list(range(2005, 2018))

    data_dir = ''

    dict_sql_default = dict(sc='profiles_raw',
                            tb='load_complete',
                            db='storage2')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
                    ('nd_id', 'VARCHAR'),
                    ('value', 'DOUBLE PRECISION'),
                    ('hy', 'SMALLINT'),
                    ('year', 'SMALLINT')]
    tb_pk = ['nd_id', 'year', 'hy']


    def __init__(self, kw_dict, tb='entsoe_load', filt=[]):
        ''' Sets up the object. Doesn't run the methods. '''

        super().__init__(**kw_dict)

        self.tb = tb
        self.filt = filt

    def read_data(self):

        self.df = aql.read_sql(self.dict_sql['db'], 'profiles_raw',
                               self.tb, filt=self.filt)


        # delete incomplete days
        self.df['doy'] = self.df.DateTime.dt.dayofyear
        slct_incomp = self.df.pivot_table(values='value',
                                          index=['year', 'doy'],
                                          aggfunc=len)
        slct_incomp = slct_incomp.loc[pd.DataFrame(slct_incomp)['value'] < 24].index.values
        self.df = self.df.loc[-self.df[['year', 'doy']]
                                   .apply(tuple, axis=1)
                                   .isin(slct_incomp)]

        # delete
        self.df = self.filter_leap_day(self.df)


        # there must be only one nd_id
        self.nd_id = self.df.nd_id.unique()[0]




    def add_time_cols(self):

        self.df_tot['year'] = self.df_tot.DateTime.dt.year
        self.df_tot['dow'] = self.df_tot.DateTime.dt.dayofweek
        self.df_tot['wk'] = self.df_tot.DateTime.dt.week
        self.df_tot['hour'] = self.df_tot.DateTime.dt.hour
        self.df_tot['mt'] = self.df_tot.DateTime.dt.month
        self.df_tot['day'] = self.df_tot.DateTime.dt.day


    def normalize_value_column(self):
        ''' Normalizes the value column to one for each year separately. '''

        self.df_tot['value'] = (self.df_tot.groupby('year')['value']
                                           .apply((lambda x: x/x.sum())))

    def init_df_tot(self):
        '''
        Merge the timemap for all years with the input data.
        This yields the total table with the gaps to be filled subsequently.
        '''

        self.df_tot = (self.tm.df_time_map[['DateTime', 'dow_name']]
                              .join(self.df.set_index('DateTime'),
                                    on='DateTime'))
        self.df_tot = self.df_tot.reset_index(drop=True)

        self.normalize_value_column()

        self.df_tot['nd_id'] = self.nd_id

        self.add_time_cols()

        # if the year's last partial week is equal to 1,
        # set its value to max + 1;
        # likewise, if the year's first partial week is in [52, 53],
        # set its value to 0
        def fix_last_week_index(df):
            if df.iloc[0]['mt'] == 12 and 1 in df.wk.values:
                df.loc[df.wk == 1, 'wk'] = df.wk.max() + 1
            if df.iloc[0]['mt'] == 1 and (df.wk > 25).sum() > 0:
                df.loc[df.wk > 25, 'wk'] = 0
            return df
        self.df_tot = (self.df_tot.groupby(['year', 'mt'])
                                  .apply(fix_last_week_index))


    def init_df_pv(self):

        # pivot tables drop all-nan columns. therefore we set inexistant
        # values to 0. then the nan values in the pivot table are
        # invalid dates which can be disregarded.
        self.df_tot['value'] = self.df_tot.value.fillna(0)

        pv_kwargs = dict(values='value', index=['wk', 'dow', 'hour'],
                         columns='year')

        if self.df_tot.pivot_table(**pv_kwargs, aggfunc=len).max().max() > 1:
            raise ValueError('pivot_table aggregates multiple values.')
        self.df_pv = self.df_tot.pivot_table(**pv_kwargs, aggfunc=np.sum)

        pv_day_kwargs = dict(values='value', index=['wk', 'dow'], columns='year')
        self.df_pv_day = self.df_tot.pivot_table(**pv_day_kwargs, aggfunc=np.sum)

    def get_missing_years(self):
        df_max = self.df_pv.max()
        self.list_old = (df_max.loc[df_max > 0]
                               .index.get_level_values('year').tolist())
        self.list_new = (df_max.loc[df_max == 0]
                               .index.get_level_values('year').tolist())

    def fill_incomplete_by_day(self):
        '''
        For each day, select the existing year X which contains thee median
        day by total energy. Then copy the corresponding day profile of year X
        to all years where this day is missing.
        '''

        def get_median_day(x):
            # Note: not using median function to avoid averaging
            x_sort = x.loc[-x.isnull()].sort_values()
            med_yr = (x_sort.reset_index()
                            .iloc[int(len(x_sort) // 2)]
                            .loc['year']
                            .values[0])
            return med_yr

        self.df_pv_day['median_year'] = (self.df_pv_day[self.list_old]
                                             .apply(get_median_day, axis=1))

        dict_day_year = self.df_pv_day['median_year'].to_dict()

        # add per-day median year selection to df_pv
        self.df_pv = (self.df_pv.reset_index()
                                .join(self.df_pv_day['median_year'],
                                      on=self.df_pv_day.index.names)
                                .set_index(self.df_pv.index.names))

        # switching from nan to inf for invalid datetimes so we can use fillna
        # on the missing data
        self.df_pv[self.df_pv.isnull()] = np.inf
        self.df_pv[self.df_pv == 0] = np.nan

        fill_slct_day = lambda x: x.T.fillna(x[dict_day_year[x.name]].T).T
        self.df_pv = (self.df_pv.groupby(level=['wk', 'dow'])
                                .apply(fill_slct_day))


    def finalize_table(self):

        # stack final table by years
        self.dfnew = self.df_pv[self.list_old + self.list_new].stack().reset_index()
        self.dfnew.columns = self.df_pv.index.names + ['year', 'value']

        # remove invalid datetimes
        self.dfnew = self.dfnew.loc[-(self.dfnew.value == np.inf)]

        # add dfnew to df_tot
        self.df_tot = self.df_tot.rename(columns={'value': 'value_orig'}).join(self.dfnew.set_index(self.df_pv.index.names + ['year']), on=self.df_pv.index.names+ ['year'])

        self.df_tot['DateTime'] = self.df_tot.DateTime.dt.tz_localize('UTC')

        self.df_tot = self.get_hour_of_the_year(self.df_tot[['DateTime', 'nd_id', 'value', 'value_orig', 'year']])

        # re-add the time index columns which were lost due to the hy generation
        self.add_time_cols()

        # re-normalize all years to one
        self.normalize_value_column()

        print('Writing final table to database...', end='')
        self.append_to_sql(self.df_tot[[c for c in self.df_tot.columns
                                        if c in [col[0].replace('"', '')
                                        for col in self.tb_cols]]].copy())
        print('done.')


# %%

if __name__ == '__main__':
    pya = ProfileYearAdder('', [])#, filt=[('nd_id', ['DE0'])])
    self = pya

    for tb, filt in [('swissgrid_load', []),
                     ('terna_load', []),
                     ('rte_load', []),
                     ('econtrol_load', []),
                     ('entsoe_load', [('nd_id', ['DE0'])])]:

        self.tb = tb
        self.filt = filt

        self.read_data()
        self.init_tm(self.all_yr)
        self.init_df_tot()
        self.init_df_pv()
        self.get_missing_years()
        self.fill_incomplete_by_day()
        self.finalize_table()

# %%
if __name__ == '__main__':

    df = aql.read_sql('storage2', 'profiles_raw', 'entsoe_load', filt=[('nd_id', ['DE0'])])
    df.pivot_table(index='hy', values='value', columns='year', aggfunc=sum).plot()
    df.year.drop_duplicates()

    df = aql.read_sql('storage2', 'profiles_raw', 'terna_load')
    df.pivot_table(index='hy', values='value', columns='year', aggfunc=sum).plot()
    df.year.drop_duplicates()

    df = aql.read_sql('storage2', 'profiles_raw', 'econtrol_load')
    df.pivot_table(index='hy', values='value', columns='year', aggfunc=sum).plot()
    df.year.drop_duplicates()

    df = aql.read_sql('storage2', 'profiles_raw', 'rte_load')
    df.pivot_table(index='hy', values='value', columns='year', aggfunc=sum).plot()
    df.year.drop_duplicates()

    df = aql.read_sql('storage2', 'profiles_raw', 'swissgrid_load')
    df.pivot_table(index='hy', values='value', columns='year', aggfunc=sum).plot()
    df.year.drop_duplicates()


# %%
reload(profile_reader)

class EntsoeLoadReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='entsoe_load')
    data_dir = os.path.normpath('DEMAND/ENTSOE/')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('nd_id', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('year', 'SMALLINT')]
    tb_pk = ['nd_id', 'year', 'hy']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read(self, fn):

        df_add = pd.read_csv(fn).fillna(0)

        df_add['DateTime'] = pd.to_datetime(df_add['Time (UTC)'].apply(lambda x: x.split(' - ')[0]))
        df_add['DateTime'] = df_add['DateTime'].dt.tz_localize('UTC')

        df_add = df_add.iloc[:, [2, -1]]
        df_add.columns = ['value', 'DateTime']
        df_add['nd_id'] = fn.split(os.path.sep)[-1].split('_')[0] + '0'

        df_add = self.time_resample(df_add, res='H')

        return df_add

    def post_processing(self, df):

        return self.filter_years_by_data_length(df)

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   exclude_substrings=[],
                   col_filt=[],
                   ext=['csv'])

    op = EntsoeLoadReader(kw_dict)

    self = op

    fn = self.fn_list[10]

    self.read_all()

    self.df_tot.loc[self.df_tot.nd_id == 'DE0'].pivot_table(values='value', index='hy', columns=['nd_id', 'year']).plot()

reload(profile_reader)

# %%

reload(profile_reader)

class TernaProfileReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='terna_load')
    data_dir = os.path.normpath('DEMAND/ITALY/')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('nd_id', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('year', 'SMALLINT')]
    tb_pk = ['nd_id', 'year', 'hy']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()
        self.date_list = []
        self.date_fn_list = []

        yr_list = list(set([int(yy.split('.')[0].split(os.path.sep)[-1].split('_')[0].replace('Confronto', ''))
               for yy in self.fn_list_0]))

        self.init_tm(yr_list) # needs TimeMap for dst hours fixing


    def get_fn_list(self):
        '''
        Get htm files, parse the urls to the PDFs, download the PDFs and
        save the file names in the fn_list attribute.

        https://www.terna.it/it-it/sistemaelettrico/dispacciamento/datiesercizio/datigiornalieri/confronto.aspx
        '''

        super().get_fn_list()

        # switching from htm file list to pdf file list
        self.fn_list_0 = self.fn_list.copy()

        # get complete list of urls
        url_list = [url for fn0 in self.fn_list_0 for url in self.get_urls(fn0)]

        self.fn_list = self.download_files(url_list)


    def get_urls(self, fn):

        files = []

        soup = BeautifulSoup(open(fn))
        links = soup.find_all('a')

        for tag in links:
            link = tag.get('href',None)
            if link is not None:
                if 'download.terna' in link:
                    files.append(link)

        return files


    def download_files(self, file_list):
        fn_w_list = []
        for download_url in tqdm(file_list):

            fn_w = os.path.join(self._dir, '_'.join(download_url.split('/')[-2:]))

            if not os.path.exists(fn_w):

                response = urllib.request.urlopen(download_url)
                file = open(fn_w, 'wb')
                file.write(response.read())
                file.close()

            fn_w_list.append(os.path.join(self._dir, fn_w))

        return fn_w_list

    def parse_pdf(self, fn):
        '''
        Seems to require different styles depending on the year.

        date and time formats change over the years.
        '''

        pdfFileObj = open(fn, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pageObj = pdfReader.getPage(0)
        str_text = pageObj.extractText()

        time = np.array([])
        cons = np.array([])
        prev = np.array([])

        cc = 0

        dict_ital_months = {'dic': '12', 'nov': '11', 'ott': '10', 'set': '09',
                            'ago': '08', 'lug': '07', 'giu': '06', 'mag': '05',
                            'apr': '04', 'mar': '03', 'feb': '02', 'gen': '01'}

        # replace italian months with month number
        repl_months = lambda x: [x.replace(kk, vv)
                                 for kk, vv in dict_ital_months.items()
                                 if kk in x][0]

        file_code = str_text[:6]
        print('File code: ', file_code)

        # skip invalid files
        if file_code in ['CDP Re']:
            return None



        if file_code in ['*OraPr', 'OraPre']:
            date = str_text[str_text.find(', ') + 2:
                            str_text.find('Confronto')]

            date = repl_months(date)
            day, month, year = [int(dd) for dd in date.split(' ')]


            data = str_text[str_text.find('eConsuntivo') + 11:
                            str_text.find('Scostamento Assoluto')]

            for ih in range(1,25):

                nhour = len(str(ih)) # number of digits of the hour

                # note: hours start at 1, hence shifted by one hour
                time = np.append(time, int(data[cc:cc + nhour]) - 1)
                cc += nhour
                cons = np.append(cons, data[cc:cc + 6].replace(',','').replace('.',''))
                cc += 6
                prev = np.append(prev, data[cc:cc + 6].replace(',','').replace('.',''))
                cc += 6

        if file_code == 'Report':
            date = str_text[str_text.find(' : ') + 3:
                            str_text.find(' : ') + 13]
            day, month, year = [int(dd) for dd in date.split('/')]

            ind_italia = str_text.find('ITALIA')

            if ind_italia > -1:
                data = str_text[str_text.find('ITALIA') + 6:]
            else:
                data = str_text[str_text.find('100%PREVISIONEORACONSUNTIVOGiorno di previsione :') +50: ]
            data = data.replace(date, '')
            data = data.replace(',', '')

            len_num = data.find(':') - 2

            for ih in range(1,25):
                cons = np.append(cons, data[cc:cc + len_num].replace(',','').replace('.',''))
                cc += len_num
                # note: hours start at 1, hence shifted by one hour
                time = np.append(time, int(data[cc:cc + 5].split(':')[0]) - 1)
                cc += 5
                prev = np.append(prev, data[cc:cc + len_num].replace(',','').replace('.',''))
                cc += len_num

        print(date)

        df_day = pd.DataFrame(np.stack([time, cons]).T, columns=['hour', 'value'])
        for nn, tt in [('year', year), ('month', month), ('day', day)]:
            df_day[nn] = tt

        return df_day, date

    def postprocessing_tot(self):
        '''
        Various operations once the table df_tot has been assembled.
        '''
        self.fix_double_dates()
        self.fix_missing_dst()
        self.df_tot = self.filter_years_by_data_length(self.df_tot)
        self.tz_localize_convert(tz='Europe/Rome')
        self.df_tot = self.get_hour_of_the_year(self.df_tot[['DateTime', 'value', 'nd_id']])
        self.append_to_sql(self.df_tot.copy())

    def read(self, fn):

        df_add, date = self.parse_pdf(fn)

        self.date_fn_list.append((date, fn))
        self.date_list.append(date)

        if df_add is None:
            return

        df_add['DateTime'] = pd.to_datetime(df_add[['hour', 'month', 'year', 'day']])
        df_add['value'] = df_add['value'].astype(float)

        df_add = df_add[['DateTime', 'year', 'value']]
        df_add['nd_id'] = 'IT0'

        return df_add

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   exclude_substrings=[],#[str(c) for c in range(2000, 2020) if not c == 2017],
                   col_filt=[],
                   ext=['htm', 'html'])

    op = TernaProfileReader(kw_dict)

    self = op
    fn = self.fn_list[1]

    self.read_all(skip_sql=True)
    self.postprocessing_tot()


# %%
#
#self.df_tot.loc[(self.df_tot.year == 2013)
#              & (self.df_tot.DateTime.dt.month == 12)].pivot_table(index=self.df_tot.DateTime.dt.day, values='value', aggfunc=len)

# %%


class RTELoadReader(profile_reader.ProfileReader):
    '''
    Read load data from RTE Historique_consommation_INST_*.xls files,
    obtained from http://clients.rte-france.com/lang/fr/visiteurs/vie/vie_stats_conso_inst.jsp
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='rte_load')
    data_dir = os.path.normpath('DEMAND/FRANCE')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('nd_id', 'VARCHAR'),
               ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'),
               ('year', 'SMALLINT')]
    tb_pk = ['nd_id', 'year', 'hy']

    exclude_substrings=[]

    def __init__(self, kw_dict):
        print(kw_dict)
        super().__init__(**kw_dict)

        self.get_fn_list()

        yr_list = [int(yy.split('.')[0].split(os.path.sep)[-1].split('_')[-1])
                   for yy in self.fn_list]
        self.init_tm(yr_list) # needs TimeMap for dst hours fixing


    def read(self, fn):
        '''
        Read RTE data from xls files.

        Note: DST switch in March shows doubled values for hours 2-3.
        '''

        df_add = pd.read_excel(fn, skiprows=18).dropna()

        df_add = df_add.drop('Type de données', axis=1).set_index('Date')
        df_add = (df_add.stack().reset_index()
                        .rename(columns={'level_1': 'Time', 0: 'value'}))


        df_add.loc[df_add['Time'] == '24:00', 'Time'] = '00:00'

        df_add['DateTime'] = df_add[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
        df_add['DateTime'] = pd.to_datetime(df_add['DateTime'], format='%d/%m/%Y %H:%M')

        df_add.loc[df_add['Time'] == '00:00', 'DateTime'] += pd.Timedelta(1, 'd')

        df_add = df_add.drop(['Date', 'Time'], axis=1)

        df_add['nd_id'] = 'FR0'

        return df_add[['DateTime', 'value', 'nd_id']]

    def postprocessing_tot(self):
        '''
        Various operations once the table df_tot has been assembled.
        '''

        self.fix_missing_dst()
        self.df_tot = self.filter_years_by_data_length(self.df_tot)
        self.tz_localize_convert(tz='Europe/Paris')
        self.df_tot = self.time_resample(self.df_tot[['DateTime', 'value', 'nd_id']])
        self.df_tot = self.get_hour_of_the_year(self.df_tot)
        self.append_to_sql(self.df_tot.copy())

if __name__ == '__main__':

    dict_sql = dict(db='storage2')
    kw_dict = dict(dict_sql=dict_sql, col_filt=[], ext='xls')
    op = RTELoadReader(kw_dict)

    self = op
    fn = self.fn_list[-1]

    # note: due to conversion to UTC not there is some overlap between the years
    # therefore we cannot write directly
    op.read_all(skip_sql=True)
    self.postprocessing_tot()





# %%



class EControlLoadReader(profile_reader.ProfileReader):
    '''
    Read load data from e-control Belastungsablauf_Oeff_*.xlsx files,
    obtained from https://www.e-control.at/statistik/strom/betriebsstatistik/betriebsstatistik2014
    etc, as well as from google filename.
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='econtrol_load')
    data_dir = os.path.normpath('DEMAND/AUSTRIA')

    tb_cols = [('"DateTime"', 'TIMESTAMP'),
               ('nd_id', 'VARCHAR'), ('value', 'DOUBLE PRECISION'),
               ('hy', 'SMALLINT'), ('year', 'SMALLINT')]
    tb_pk = ['nd_id', 'year', 'hy']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()

    def read(self, fn):
        '''
        Reads input files and generates a dataframe.

        Time zone is Europe/Vienna. Conversion to UTC and comparison to
        UTC entso-e data yield satisfactory results.
        '''

        df_add = pd.DataFrame()
        for imt in range(1,13):
            print('Reading month {}'.format(imt))

            df_addp = pd.read_excel(fn, sheet_name=format(imt, '02'), usecols='K:M', header=23).dropna()

            df_add = pd.concat([df_add, df_addp])

        df_add.columns = ['DateTime', 'energy', 'value']
        df_add = df_add.drop('energy', axis=1)

        df_add['DateTime'] = pd.to_datetime(df_add['DateTime'].apply(lambda x: x.split('-')[0]), format='%d.%m.%y   %H:%M')

        df_add['DateTime'] = df_add['DateTime'].dt.tz_localize('Europe/Vienna', ambiguous='infer')
        df_add['DateTime'] = df_add['DateTime'].dt.tz_convert('UTC')

        df_add['nd_id'] = 'AT0'


        df_add = self.time_resample(df_add)

        return df_add

    def postprocessing_tot(self):

        self.df_tot['year'] = self.df_tot.DateTime.dt.year
        self.df_tot = self.filter_years_by_data_length(self.df_tot)
        self.df_tot = self.get_hour_of_the_year(self.df_tot)
        self.append_to_sql(self.df_tot)

if __name__ == '__main__':

    dict_sql = dict(db='storage2')

    kw_dict = dict(dict_sql=dict_sql,
                   col_filt=[], ext='xlsx')

    op = EControlLoadReader(kw_dict)
    self = op
    fn = self.fn_list[1]
    op.read_all(skip_sql=True)
    op.postprocessing_tot()




# %%

class SwissGridLoadReader(profile_reader.ProfileReader):
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




