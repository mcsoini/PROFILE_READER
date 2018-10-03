# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:10:15 2018

@author: ashreeta
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import grimsel.auxiliary.timemap as timemap
from PROFILE_READER.profile_reader import ProfileReader

svg_file = "/mnt/data/Dropbox/SHARED_DATA/AGORA_SVGS"

fn = 'erzeugung_verbrauch_2012_12.svg'
svg_file = os.path.join(svg_file, fn)
from xml.dom import minidom


class AgoraReader(ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='agora_profiles',
                            db='storage2')
    data_dir = os.path.normpath('AGORA_SVGS')

    # sql table columns
    tb_cols = [('"DateTime"', 'TIMESTAMP'),
                ('nd_id', 'VARCHAR'),
                ('fl_id', 'VARCHAR'),
                ('value', 'DOUBLE PRECISION'),
                ('year', 'SMALLINT'),
#                ('hy', 'SMALLINT')
                ]
    # sql table primary key
    tb_pk = ['fl_id', '"DateTime"']


                                  # '''  lignite demand '''
    dict_scale_shift = {('2012', '01'): (15.247, 52.111),
                        ('2012', '10'): (20.917, 58.786),  # monday after last sunday
                        ('2013', '01'): ( 9.439, 52.782),
                        ('2013', '10'): (11.533, 53.895),  # monday after last sunday
                        ('2014', '01'): (16.869, 51.914),
                        ('2014', '10'): (19.626, 54.932),  # monday after last sunday
                        ('2015', '01'): (17.526, 54.423),
                        ('2015', '10'): (18.430, 53.173),  # monday after last sunday
                        ('2016', '01'): (15.727, 53.277),
                        ('2016', '10'): (18.313, 52.303),  # monday after last sunday
                        ('2017', '01'): (13.854, 52.841),
                        ('2017', '10'): (8.187,  53.112),  # monday after last sunday
                        }

    dict_fl = {
              ('2012', '10'): {56: 'wind_offshore', 54: 'photovoltaics',
                               49: 'pumped_hydro', 57: 'hydro_total',
                               48: 'others', 58: 'bio_all',
                               55: 'wind_onshore', 50: 'natural_gas',
                               51: 'hard_coal', 53: 'nuclear_fuel',
                               52: 'lignite', 59: 'dmnd'},
              ('2013', '10'): {53: 'wind_offshore', 51: 'photovoltaics',
                               46: 'pumped_hydro', 45: 'others',
                               54: 'hydro_total', 48: 'hard_coal',
                               55: 'bio_all', 47: 'natural_gas',
                               50: 'nuclear_fuel', 49: 'lignite',
                               52: 'wind_onshore', 56: 'dmnd'},
              ('2014', '10'): {51: 'wind_offshore', 49: 'photovoltaics',
                               44: 'pumped_hydro', 43: 'others',
                               52: 'hydro_total', 53: 'bio_all',
                               50: 'wind_onshore', 45: 'natural_gas',
                               46: 'hard_coal', 48: 'nuclear_fuel',
                               47: 'lignite', 54: 'dmnd'},
              ('2015', '10'): {52: 'photovoltaics', 55: 'hydro_total',
                               47: 'pumped_hydro', 54: 'wind_offshore',
                               53: 'wind_onshore', 46: 'others',
                               56: 'bio_all', 48: 'natural_gas',
                               51: 'nuclear_fuel', 49: 'hard_coal',
                               50: 'lignite', 57: 'dmnd'},
              ('2017', '10'): {51: 'photovoltaics',
                               52: 'wind_onshore', # 53
                               50: 'nuclear_fuel',
                               49: 'lignite', #52
                               48: 'hard_coal', # 53
                               47: 'natural_gas',
                               44: 'others',# 46
                               53: 'wind_offshore', # 54
                               54: 'hydro_total', #55
                               46: 'pumped_hydro',#53
                               55: 'bio_all', #56
                               56: 'dmnd'},
                ('all', '01'): {40: 'others', 41: 'pumped_hydro',
                               42: 'natural_gas', 43: 'hard_coal',
                               44: 'lignite', 45: 'nuclear_fuel',
                               46: 'photovoltaics', 47: 'wind_onshore',
                               48: 'wind_offshore', 49: 'hydro_total',
                               50: 'bio_all', 51: 'dmnd'},
                ('2017', '01'): {49: 'photovoltaics', # 46
                                 53: 'bio_all', # 50
                                 52: 'hydro_total', #49
                                 50: 'wind_onshore', #47
                                 48: 'nuclear_fuel', #45
                                 47: 'lignite', #44
                                 46: 'hard_coal', #43
                                 45: 'natural_gas', #42
                                 44: 'pumped_hydro', #41
                                 42: 'others', #40

                                 40: 'wind_offshore', # 48
                                 51: 'dmnd'},
              ('2016', '10'): {54: 'photovoltaics',
                               52: 'lignite',
                               53: 'nuclear_fuel',
                               58: 'bio_all',
                               56: 'wind_offshore',
                               50: 'natural_gas',
                               48: 'others',
                               55: 'wind_onshore',
                               51: 'hard_coal',
                               49: 'pumped_hydro',
#                               57: 'dmnd',
                               59: 'hydro_total',
                               },

#              ('2016', '10'): {54: 'photovoltaics',
#                               : 'hydro_total',
#                               : 'pumped_hydro',
#                               : 'wind_offshore',
#                               : 'wind_onshore',
#                               : 'others',
#                               : 'bio_all',
#                               : 'natural_gas',
#                               : 'nuclear_fuel',
#                               : 'hard_coal',
#                               : 'lignite',
#                               : 'dmnd'}
                }




#self.df_tot_mod.loc[(self.df_tot_mod.DateTime > '2016-12-19 23:00:00+00:00')
#                    ].pivot_table(columns='fl_id', index='DateTime', values='value').plot.area()
#
#self.df_tot_mod.loc[(self.df_tot_mod.DateTime > '2016-12-20 00:00:00+01:00')].pivot_table(values='value',
#                                                                                        columns='fl_id',
#                                                                                        index='DateTime').abs().plot.area()


    def __init__(self, kw_dict):

        super().__init__(**kw_dict)

        self.last_row = None

        self.dict_fl.update({(str(year), '01'): self.dict_fl[('all', '01')]
                         for year in range(2012, 2018)
                         if not (str(year), '01') in self.dict_fl.keys()})

    def single_post_processing(self, df):
        return df

    def read(self, fn):

        # %
        doc = minidom.parse(fn) # parseString also exists
        path_strings = [path.getAttribute('d') for path
                        in doc.getElementsByTagName('path')]
        doc.unlink()

        yr, mt = fn.split(os.path.sep)[-1].split('.')[0].split('_')[-2:]

        # some pre-filtering to get the relevant (i.e. long) paths
        list_path = [(npath, len(path),)
                     for npath, path in enumerate(path_strings)
                     if len(path) > 200]

        # init dataframes to hold all series of the current month
        df_tot = pd.DataFrame()
        df_tot_dmnd = pd.DataFrame()

#        path = [pp for pp in list_path if pp[0] == 51][0]
        for path in list_path:

            # get dataframe from path
            df = pd.DataFrame([pp.replace('M ', '').split(' ')
                               for pp in path_strings[path[0]].split(' L ')])
#            df.applymap(float).plot.scatter(x=0, y=1, linewidth=1)
            df = df.applymap(float).set_index(0)[1]

            df = df.reset_index()
            df.loc[df[0].shift(1) <= df[0], 'dir'] = 'lr'
            df.loc[df[0].shift(1) >= df[0], 'dir'] = 'rl'
            df['dir'] = df['dir'].fillna('lr')

            df_stack = df.pivot_table(values=1, index=0, columns='dir')

            df_stack[path[0]] = df_stack.rl - df_stack.lr if 'rl' in df_stack.columns else df_stack.lr

            df_stack = df_stack.reset_index(drop=True)

            if 'rl' in df_stack.columns:
#                df_stack[path[0]].plot.area()
                df_tot = pd.concat([df_tot, df_stack[path[0]]], axis=1)
            else:
                df_tot_dmnd = pd.concat([df_tot_dmnd, df_stack[path[0]]], axis=1)


        df_tot = df_tot.T.drop_duplicates().T

        df_tot_dmnd = df_tot_dmnd.T.drop_duplicates().T


        mtf = int(mt.replace('a', '').replace('b', ''))
        yrf = int(yr)
        yr_next = yrf if mtf < 12 else yrf + 1
        mt_next = mtf + 1 if mtf < 12 else 1
        dftm = pd.DataFrame(index=pd.date_range('{}-{:02d}-01'.format(yrf, mtf),
                                                '{}-{:02d}-01 23:30:00'.format(yr_next, mt_next),
                                                tz='Europe/Berlin', freq='H'))

        dftm = dftm.reset_index().rename(columns={'index': 'DateTime'})

        # get last sunday
        last_sunday = dftm.loc[(dftm.DateTime.dt.dayofweek == 6)
                               & (dftm.DateTime.dt.month == mtf), 'DateTime'].dt.date.iloc[-1]

        if '10a' in fn:
            # month ends with 'last sunday'
            dftm = dftm.loc[dftm.DateTime.dt.date < last_sunday]
            dftm = dftm.reset_index(drop=True)

        elif '10b' in fn:
            # month begins after 'last sunday'
            dftm = dftm.loc[dftm.DateTime.dt.date > last_sunday]
            dftm = dftm.reset_index(drop=True)


        df_tot = pd.concat([dftm, df_tot, df_tot_dmnd], axis=1)

        df_tot = df_tot.set_index('DateTime')


        df_tot = df_tot.stack().reset_index()

        df_tot = df_tot.rename(columns={'level_0': 'DateTime', 'level_1': 'fl_id', 0: 'value'})

        df_tot['mt'] = mt
        df_tot['year'] = yr

        # %

        return df_tot

    def align_and_scale(self, df, year):

        #%

        print('?'*60 + '\n', year)

        df['dtday'] = df.DateTime.dt.day.apply(lambda x: '{:02d}'.format(x))
        df['dtmonth'] = df.DateTime.dt.month.apply(lambda x: '{:02d}'.format(x))
        df['dtyear'] = df.DateTime.dt.year.apply(str)
        df['dtdate'] = df[['dtyear', 'dtmonth', 'dtday']].apply(lambda x: ' '.join(x), axis=1)

        df = df.sort_values(['DateTime'])

        # NOTE: mt is month from file name
        list_mt = df['mt'].drop_duplicates().values
        dfg = df.groupby(['mt'])

        # get last row of first month ever
        dfg_slct = dfg.get_group(list_mt[0])

        dfg_slct.loc[:, 'fl_id'] = dfg_slct.fl_id.replace(self.dict_fl[(year, '01')])
        scale_power = (self.dict_scale_shift[(str(year), '01')][0]
                       / dfg_slct.loc[dfg_slct.fl_id.isin(['lignite'])].value.iloc[0])
        dfg_slct['value'] *= scale_power

        # get the last date, which should be the first date in the next file
        last_date = dfg_slct.dtdate.iloc[-1]

        # get the average production of the last day, to be matched with the next
        sum_last_day = dfg_slct.loc[dfg_slct.dtdate == last_date]
        sum_last_day = sum_last_day.pivot_table(index='fl_id', values='value')

        df_tot_new = dfg_slct.copy()

        # start with second month ever
        mt = list_mt[1]
        for mt in list_mt[1:]:
            print('$'*60 + '\n', mt)
#
            dfg_slct = dfg.get_group(mt)

            if not mt == '10b':
                first_date = last_date
            else:
                first_date = dfg_slct.dtdate.iloc[0]

            # get the average production of the last day, to be matched with the next
            sum_first_day = dfg_slct.loc[dfg_slct.dtdate == first_date]
            sum_first_day= sum_first_day.pivot_table(index='fl_id', values='value')


            df_fl_map = pd.concat([sum_first_day.sort_values('value').reset_index(),
                                   sum_last_day.sort_values('value').reset_index().rename(columns={'fl_id': 'fl_last', 'value': 'value_last'})
                                  ], axis=1)
            df_fl_map ['scale_power'] = df_fl_map['value_last'] / df_fl_map.value

            print(df_fl_map)
            if not mt == '10b':
                dict_fl_map = df_fl_map.set_index('fl_id')['fl_last'].to_dict()
            else:
                dict_fl_map = self.dict_fl[(year, '10')]

                # map fl in dfg_slct
            dfg_slct.loc[:, 'fl_id'] = dfg_slct.fl_id.replace(dict_fl_map)


            if not mt == '10b':
                # no valid scaling factor calculated for 10b
                scale_power = df_fl_map.scale_power.mean()
            else:
                scale_power = (self.dict_scale_shift[(str(year), '10')][0]
                               / dfg_slct.loc[dfg_slct.fl_id.isin(['lignite'])].value.iloc[0])

            # scale, assuming the scaling factor doesn't change
            dfg_slct['value'] *= scale_power

            df_tot_new = df_tot_new.loc[-df_tot_new.dtdate.isin([last_date])]
            df_tot_new = pd.concat([df_tot_new, dfg_slct])

            # redefine last_date
            last_date = dfg_slct.dtdate.iloc[-1]

            # get the average production of the last day, to be matched with the next
            sum_last_day = dfg_slct.loc[dfg_slct.dtdate == last_date]
            sum_last_day = sum_last_day.pivot_table(index='fl_id', values='value')

        # %

#        df_tot_new = df_tot_new.drop_duplicates()

        # shift demand
        df_tot_new.loc[df_tot_new.fl_id == 'dmnd', 'value'] *= -1

        shift_dmnd = (self.dict_scale_shift[(str(year), '01')][1]
                      - df_tot_new.loc[df_tot_new.fl_id == 'dmnd', 'value'].iloc[0])


        df_tot_new.loc[df_tot_new.fl_id == 'dmnd', 'value'] += shift_dmnd


        return df_tot_new

    def postprocessing_tot(self):

        dftotg = self.df_tot.copy().groupby('year')

        self.df_tot_mod = pd.DataFrame()

        iyr = '2017'
        for iyr in list(dftotg.groups.keys()):
            df = dftotg.get_group(iyr)
            year = iyr
            print(year, iyr)

            df_add = self.align_and_scale(df, year)
            df_add = df_add.loc[df_add.DateTime.dt.year == int(year)]
            self.df_tot_mod = pd.concat([self.df_tot_mod, df_add])

        self.df_tot_mod['DateTime'] = self.df_tot_mod.DateTime.dt.tz_convert('UTC')

        self.df_tot_mod = self.df_tot_mod[[c for c in self.df_tot_mod
                                           if not (c in
                                           ['year', 'mt'] or 'dt' in c)]]

        self.df_tot_mod.loc[:, 'year'] = self.df_tot_mod.DateTime.dt.year
#        self.df_tot_mod = self.get_hour_of_the_year(self.df_tot_mod)

        self.df_tot_mod = self.filter_years_by_data_length(self.df_tot_mod)

        self.df_tot_mod['nd_id'] = 'DE0'
#
#        self.df_tot_mod = (self.df_tot_mod.set_index('DateTime')
#                                          .drop_duplicates()
#                                          .reset_index())
#
        for df in self.df_tot_mod.groupby(['fl_id']):
            print(df[1].iloc[:3])
            self.append_to_sql(df[1])
#

#        from grimsel_h.auxiliary.aux_general import print_full
#        print_full(
#        self.df_tot_mod.loc[self.df_tot_mod.fl_id.isin(['hard_coal'])
#                            & (self.df_tot_mod.DateTime >= '2012-12-31 00:00:00')
#                            & (self.df_tot_mod.DateTime <= '2013-01-02 23:00:00')].sort_values('DateTime')
#        )
#
        # %

if __name__ == '__main__':
    
    kw_dict = dict(dict_sql=dict(db='storage2'),
                   tm_filt={'year': range(2005, 2018)},
                   col_filt=[], ext='svg', exclude_substrings=['ch_2'])
    
    op = AgoraReader(kw_dict)
    
    op.get_fn_list()
    
    self = op
    
    fn = op.fn_list[1]
    fn = [f for f in op.fn_list if '2017_02' in f][0]
    op.read_all(skip_sql=True)

    op.postprocessing_tot()




#%%
if __name__ == '__main__':
    
    #
    #self.df_tot.loc[self.df_tot.year.isin(['2016'])
    #                & (self.df_tot.DateTime > '2016-10-10 00:00:00+02:00')].pivot_table(columns='fl_id', index='DateTime', values='value').plot.area()
    
    
    do = pltpg.PlotPageData.from_df(df=self.df_tot_mod.loc[self.df_tot_mod.year.isin(['2016', '2017'])
                    & (self.df_tot_mod.DateTime > '2016-10-10 00:00:00+02:00')
                    & -(self.df_tot_mod.fl_id.isin(['dmnd', 'hydro_total', 'hard_coal']))],
                                    ind_pltx=[], ind_plty=[], ind_axx=['DateTime'],
                                    series=['fl_id'], values='value')
    
    layout_kw = {'left': 0.05, 'right': 0.875, 'wspace': 0.2, 'hspace': 0.2, 'bottom': 0.1}
    label_kw = {'label_format':' {:.3f}', 'label_subset':[0], 'label_threshold':1e-6,
                'label_ha': 'left'}
    plot_kw = dict(kind_def='StackedArea', stacked=True, on_values=True, sharex=True,
                   sharey=True, linewidth=0,
                   colormap=False, barwidth=1, linecolor='k', opacitymap=1,
                   reset_xticklabels=False, legend='plots', marker='o')
    
    
    plt_0 = pltpg.PlotTiled(do, **layout_kw, **label_kw, **plot_kw)
    
    
    
    
    # %%
    
    #%%
    
    self.df_tot_mod
    
    ax = self.df_tot_mod.loc[self.df_tot_mod.fl_id.isin(['wind_onshore', 'wind_offshore', 'photovoltaics'])].pivot_table(aggfunc=sum, index='DateTime', columns='fl_id', values='value').abs().plot.area()
    #self.df_tot_mod.loc[self.df_tot_mod.fl_id == 'dmnd'].pivot_table(aggfunc=sum, index='DateTime', columns='fl_id', values='value').abs().plot(ax=ax, marker='o', color='k')
    
    # %%
    
    df = self.df_tot_mod.copy()
    df['DateTime'] = df.DateTime.dt.tz_convert('Europe/Berlin')
    
    df.loc[df.DateTime.dt.year.isin([2015])
          & df.DateTime.dt.month.isin([10])
    #      & df.DateTime.dt.day.isin([29])
    #      & (df.fl_id == 'wind_offshore')
          ].pivot_table(index='fl_id', columns='hy', values='value').iloc[:, [0, -1]]
    
    
    
    # %%
    
    data_kw = {'data_scale': 1, 'aggfunc': np.mean, 'data_threshold': 1e-10}
    indx_kw = dict(ind_axx=['DateTime'], ind_pltx=[],
                   ind_plty=['year'], series=['fl_id'],
                   values=['value'],
                   filt=[('fl_id', ['wind_onshore', 'wind_offshore', 'photovoltaics', 'dmnd'])])
    do = pltpg.PlotPageData(db=db, table='profiles_raw.agora_profiles', **indx_kw, **data_kw)
    
    
    layout_kw = {'left': 0.05, 'right': 0.875, 'wspace': 0.2, 'hspace': 0.2, 'bottom': 0.1}
    label_kw = {'label_format':' {:.2f}', 'label_subset':[0], 'label_threshold':1e-6,
                'label_ha': 'left'}
    plot_kw = dict(kind_def='StackedArea', stacked=True, on_values=False, sharex=True,
                   sharey=True,
                   colormap=False, barwidth=1, linecolor='k', opacitymap=1,
                   reset_xticklabels=False, legend='plots', marker='o')
    
    with plt.style.context(('ggplot')):
        plt_0 = pltpg.PlotTiled(do, **layout_kw, **label_kw, **plot_kw)
    
    
    
    
    # %%
    import pandas as pd
    import grimsel_h.auxiliary.aux_sql_func as aql
    from grimsel_h.auxiliary.aux_general import print_full
    import grimsel_h.plotting.plotpage as pltpg
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    
    df = pd.DataFrame(aql.exec_sql('''
            WITH tb_dmnd AS (
                SELECT "DateTime", value AS dmnd, year FROM profiles_raw.agora_profiles
                WHERE fl_id = 'dmnd'
            ), tb_vre AS (
                SELECT  "DateTime", SUM(value) AS vre, year FROM profiles_raw.agora_profiles
                WHERE fl_id LIKE 'wind%' OR fl_id LIKE 'photo%'
                GROUP BY "DateTime", year
            ), tb_mc AS (
                SELECT "DateTime", value AS mc, year FROM profiles_raw.epex_price_volumes
                WHERE nd_id = 'DE0' and quantity = 'price_eur_mwh' AND year IN (2012, 2013, 2014, 2015)
            )
            SELECT 'DE0' AS nd_id, "DateTime", year, dmnd, vre, dmnd - vre AS resload, mc fROM tb_dmnd
            NATURAL LEFT JOIN tb_vre
            NATURAL LEFT JOIN tb_mc;
            ''', db='storage2'), columns=['nd_id', 'DateTime', 'year', 'dmnd', 'vre', 'resload', 'mc'])
    
    #cmap = mpl.cm.get_cmap('Spectral') # Colour map (there are many others)
    #df.plot.scatter(x='resload', y='mc', c='year', cmap=cmap)
    
    df.loc[df.DateTime.dt.date == datetime.date(2012, 1,1)]
    
    
    data_kw = {'data_scale': 1, 'aggfunc': np.mean, 'data_threshold': 1e-10}
    indx_kw = dict(ind_axx=['resload'], ind_pltx=[],
                   ind_plty=['year'], series=['nd_id'],
                   values=['mc'])
    do = pltpg.PlotPageData.fromdataframe(df=df, **indx_kw, **data_kw)
    
    
    layout_kw = {'left': 0.05, 'right': 0.875, 'wspace': 0.2, 'hspace': 0.2, 'bottom': 0.1}
    label_kw = {'label_format':' {:.2f}', 'label_subset':[0], 'label_threshold':1e-6,
                'label_ha': 'left'}
    plot_kw = dict(kind_def='plot.line', stacked=False, on_values=False, sharex=True,
                   sharey=True, linewidth=0,
                   colormap=False, barwidth=1, linecolor='k', opacitymap=1,
                   reset_xticklabels=False, legend='plots', marker='o')
    
    with plt.style.context(('ggplot')):
        plt_0 = pltpg.PlotTiled(do, **layout_kw, **label_kw, **plot_kw)
    
    for ax in plt_0.axarr.flatten():
        ax.set_ylim([-100, 150])
    


