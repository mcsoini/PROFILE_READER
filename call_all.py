#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:57:56 2018

@author: user
"""
import os
from importlib import reload

import pandas as pd

import grimsel.auxiliary.maps as maps
import grimsel.auxiliary.aux_sql_func as aql

import PROFILE_READER.daily_gas_prices as dgp
import PROFILE_READER.various_load_readers as vlr
import PROFILE_READER.profile_reader as pr
import PROFILE_READER.sketch_agora as ag
import PROFILE_READER.hydro_level as hy
import PROFILE_READER.timemap as tm
import PROFILE_READER.monthly_production as mp

import PROFILE_READER.config as conf


os.chdir(os.path.dirname(os.path.realpath(__file__)))


# %%
#conf.

base_dir = conf.BASE_DIR

sc_maps = 'lp_input_calibration_years_linonly'
db = 'storage2'
mps = maps.Maps(sc_maps, db)


tm.build_timestamp_template(db, 'profiles_raw', 2005, 2020)



kw_dict = dict(dict_sql=dict(db=db), base_dir=base_dir,
               tm_filt={'year': range(2005, 2018)},
               col_filt=[],
               exclude_substrings=[], ext='xlsx')
op = pr.WeeklyRORReader(kw_dict)
op.read_all(skip_sql=True)
op.append_to_sql(op.df_tot)



kw_dict = dict(dict_sql=dict(db=db), base_dir=base_dir,
               tm_filt={'year': range(2005, 2018)},
               col_filt=[],
               exclude_substrings=[], ext='xlsx')
op = pr.CHPProfileReader(kw_dict)
op.read_all(skip_sql=True)
op.append_to_sql(op.df_tot)



kw_dict = dict(dict_sql=dict(db=db),  base_dir=base_dir,
               exclude_substrings=[],
               col_filt=[],
               ext=['csv'])
op = vlr.EntsoeLoadReader(kw_dict)
op.read_all()



kw_dict = dict(dict_sql=dict(db=db), exclude_substrings=[], base_dir=base_dir,
               col_filt=[], ext=['htm', 'html'])
op = vlr.TernaProfileReader(kw_dict)
op.read_all(skip_sql=True)
op.postprocessing_tot()


kw_dict = dict(dict_sql=dict(db=db), col_filt=[], ext='xls')
op = vlr.RTELoadReader(kw_dict)
# note: due to conversion to UTC not there is some overlap between the years
# therefore we cannot write directly
op.read_all(skip_sql=True)
op.postprocessing_tot()


kw_dict = dict(dict_sql=dict(db=db), col_filt=[], ext='xlsx')
op = vlr.EControlLoadReader(kw_dict)
op.read_all(skip_sql=True)
op.postprocessing_tot()


kw_dict = dict(dict_sql=dict(db=db), col_filt=[], ext='xls')
op = vlr.SwissGridLoadReader(kw_dict)
op.read_all(skip_sql=True)
op.postprocessing_tot()



reload(vlr)
kw_dict = dict(dict_sql=dict(db=db))
pya = vlr.ProfileYearAdder(kw_dict, tb='', filt=[])#, filt=[('nd_id', ['DE0'])])
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





kw_dict = dict(dict_sql=dict(db='storage2'),
               exclude_substrings=[],
               tm_filt={'year': range(2005, 2018)},
               ext=['csv'])
op = dgp.QuandlCoalPriceReader(kw_dict)
op.read_all()

kw_dict = dict(dict_sql=dict(db='storage2'),
               exclude_substrings=[],
               tm_filt={'year': range(2005, 2018)},
               ext=['xlsx'])
op = dgp.DailyGasPriceReader(kw_dict)
op.read_all()




kw_dict = dict(dict_sql=dict(db=db),
               tm_filt={'year': range(2005, 2018)},
               col_filt=[], ext='csv', exclude_substrings=[])
op = pr.EpexPriceVolume(kw_dict)
op.get_fn_list()
op.read_all()


#kw_dict = dict(dict_sql=dict(db=db),
#               tm_filt={'year': range(2005, 2018)},
#               col_filt=[], ext=['xlsx'], exclude_substrings=['Real'])
#op = pr.RTEProduction(kw_dict)
#op.get_fn_list()
#op.read_all()
kw_dict = dict(dict_sql=dict(db=db),
               tm_filt={'year': [2015, 2014, 2016]},
               col_filt=[], ext=['xlsx'], exclude_substrings=['Real'])

op = pr.RTEProduction(kw_dict)

op.get_fn_list()

op.fn_list = [f for f in op.fn_list if '2015' in f or '2016' in f or '2014' in f]

fn = op.fn_list[0]
op.read_all(skip_sql=True)
op.post_processing()


kw_dict = dict(dict_sql=dict(db=db),
           tm_filt={'year': range(2005, 2018)},
           col_filt=[])
op = pr.EntsoeGenerationReader(kw_dict)
op.get_fn_list()
op.read_all()

kw_dict = dict(dict_sql=dict(db=db),
               tm_filt={'year': range(2005, 2018)},
               col_filt=[],
               exclude_substrings=['filtered', '30', '60'])
op = pr.OpenPowerSystemReader(kw_dict)
op.read_all(skip_sql=True)
op.df_tot = op.get_hour_of_the_year(op.df_tot)
op.append_to_sql(op.df_tot)


kw_dict = dict(dict_sql=dict(db='storage2'),
               exclude_substrings=[],
               tm_filt={'year': range(2015, 2018)},
               ext=['csv'])
op = hy.FRHydroLevelReader(kw_dict)
op.read_all()




kw_dict = dict(dict_sql=dict(db='storage2'),
               tm_filt={'year': range(2005, 2018)},
               col_filt=[], ext='svg', exclude_substrings=['ch_2'])

op = ag.AgoraReader(kw_dict)
op.get_fn_list()
op.read_all(skip_sql=True)
op.postprocessing_tot()

kw_dict = dict(dict_sql=dict(db=db),
               exclude_substrings=[],
               tm_filt={'year': range(2005, 2018)},
               ext=['xlsx'])


op = mp.MonthlyProductionReader(kw_dict)
self = op
fn = self.fn_list[0]
op.read_all()


dict_sql = dict(db=db)
kw_dict = dict(exclude_substrings=['near', 'long', 'future'],
               dict_sql=dict_sql, tm_filt={'year': range(2005, 2018)},
               col_filt=[('pt_id', ['WIN_TOT'], False)])
nr = pr.NinjaReader(kw_dict)
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

cols = ['pp_id'] + [c for c in aql.get_sql_cols('plant_encar', sc_maps, db).keys()
                    if 'cap_pwr_leg' in c]
dfcap = aql.read_sql(db, sc_maps, 'plant_encar')[cols]
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

aql.write_sql(dfcap, db, 'public', 'temp_add_column', 'replace')

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
df = aql.read_sql(db, 'profiles_raw', 'ninja', filt=filt)
dfcap = aql.read_sql(db, 'public', 'temp_add_column', filt=filt)

exec_strg = '''
            SELECT pt_id, proftype, year, nd_id, location, AVG(value_sc), AVG(value)
            FROM profiles_raw.ninja
            GROUP BY pt_id, proftype, year, nd_id, location
            '''
cfnn = pd.DataFrame(aql.exec_sql(exec_strg, db=db),
             columns=['pt_id', 'proftype', 'year', 'nd_id', 'location', 'cfsc', 'cf'])

cols = ['pp_id'] + [c for c in aql.get_sql_cols('plant_encar', sc_maps, db).keys()
                    if 'cf_max' in c]
dfcf = aql.read_sql(db, sc_maps, 'plant_encar')[cols]
dfcf = dfcf.rename(columns={'cf_max': 'cf_max_yr2015'})
dfcf['pp_id'].replace(mps.dict_pp, inplace=True)
dfcf = dfcf.loc[dfcf.pp_id.str.contains('WIN|SOL')]
dfcf['nd_id'] = dfcf['pp_id'].apply(lambda x: x[:2] + '0')
dfcf['pt_id'] = dfcf['pp_id'].apply(lambda x: x[3:])
dfcf = dfcf.drop('pp_id', axis=1).set_index(['nd_id', 'pt_id']).stack().reset_index().rename(columns={'level_2': 'year', 0: 'cfmax'})
dfcf['year'] = dfcf['year'].apply(lambda x: int(x[-4:]))
dfcf = dfcf.set_index(['nd_id', 'pt_id', 'year'])

cfnn = cfnn.join(dfcf, on=dfcf.index.names)
