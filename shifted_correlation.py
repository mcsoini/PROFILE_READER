#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:16:37 2018

@author: user
"""


import matplotlib.pyplot as plt

import grimsel_h.auxiliary.aux_sql_func as aql
import pandas as pd
import numpy as np

import grimsel_h.auxiliary.timemap as timemap
#
#dfcomp = pd.DataFrame(aql.exec_sql('''
#             SELECT * FROM profiles_raw.rte_load
#             NATURAL LEFT JOIN (SELECT nd_id, value AS value_entsoe, hy, year
#             FROM profiles_raw.entsoe_load) AS ents
#             WHERE year = 2016;
#           ''', db='storage2'), columns=['nd_id', 'hy', 'year', 'DateTime',
#                                         'value', 'value_entsoe'])

#dfcomp = pd.DataFrame(aql.exec_sql('''
#            WITH tb_dmnd AS (
#                SELECT sy, nd_id, value AS dmnd, swhy_vl AS year FROM out_cal.par_dmnd
#                NATURAL LEFT JOIN (SELECT swhy_vl, run_id FROM out_cal.def_loop) AS dflp
#                NATURAL LEFT JOIN (SELECT nd_id, nd FROM out_marg_store.def_node) AS dfnd
#                WHERE nd = 'DE0' AND swhy_vl = 'yr2016'
#            ), tb_price AS (
#                SELECT sy, nd_id, value AS price, swyr_vl AS year FROM out_marg_store.par_priceprof
#                NATURAL LEFT JOIN (SELECT swyr_vl, run_id FROM out_marg_store.def_loop) AS dflp
#                NATURAL LEFT JOIN (SELECT nd_id, nd FROM out_marg_store.def_node) AS dfnd
#                WHERE nd = 'DE0' AND swyr_vl = 'yr2016'
#            )
#            SELECT sy, year, dmnd, price, nd_id FROM tb_dmnd
#            NATURAL LEFT JOIN tb_price
#           ''', db='storage2'), columns=['hy', 'year', 'value_0', 'value_1', 'nd_id'])

dfcomp = pd.DataFrame(aql.exec_sql('''
WITH tb1 AS (
    SELECT hy, year, value AS val1 FROM profiles_raw.rte_production_eco2mix
    WHERE year = 2015 AND fl_id = 'photovoltaics'
), tb0 AS (
    SELECT hy, year, value AS val0 FROM profiles_raw.entsoe_generation
    WHERE fl_id = 'photovoltaics' AND year = 2015 AND nd_id = 'FR0'
)
SELECT * FROM tb1
NATURAL LEFT JOIN tb0
''', db='storage2'), columns=['hy', 'year', 'value_0', 'value_1'])
#
#dfcomp = df_full_all[['slot', 'year', 'dmnd', 'DE_DMND_ag']].rename(columns={'slot': 'hy', 'dmnd': 'value_0', 'DE_DMND_ag': 'value_1'})
#dfcomp = dfcomp.loc[dfcomp.year.isin([2015])]



dfcomp = dfcomp.dropna(axis=0)

val_0 = dfcomp.value_0.values
val_1 = dfcomp.value_1.values

r = np.arange(-50, 51)
val_1 = np.tile(val_1, [r.shape[0], 1])
A = val_1

rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

# Use always a negative shift, so that column_indices are valid.
# (could also use module operation)
r[r < 0] += A.shape[1]
column_indices = column_indices - r[:,np.newaxis]

result = A[rows, column_indices]

df_tot = pd.DataFrame(np.concatenate((result.T, np.expand_dims(val_0, 1)), axis=1),
                      columns=[n for n in r] + ['value_0'])
df_tot = pd.concat([dfcomp[['hy', 'year']], df_tot], axis=1)

# add month indices
tm = timemap.TimeMap()
tm.gen_hoy_timemap(start='2005-1-1 00:00', stop='2017-12-31 23:59')

df_tot = df_tot.join(tm.df_time_map.set_index(['hy', 'year'])['mt'],
            on=['hy', 'year'])


df_tot = df_tot.set_index(['hy', 'year', 'value_0', 'mt']).stack().reset_index()
df_tot = df_tot.rename(columns={'level_4': 'shift', 0: 'value_1'})

# % Time shifted cross-correlation

df_corr = df_tot.groupby(['mt', 'shift']).apply(lambda df: df.corr().loc['value_1', 'value_0'])

df_corr = df_corr.reset_index()

dict_shift = {kk: kk - A.shape[1] if kk > 8000 else kk for kk in df_corr['shift'].drop_duplicates()}

df_corr['shift'].replace(dict_shift, inplace=True)

df_corr.loc[df_corr.mt.isin(['JUN', 'JUL', 'DEC', 'JAN', 'FEB', 'APR'])].pivot_table(values=0, index='shift', columns='mt').plot(marker='.')

# %% New Time shifted cross-correlation

df_tot = df_tot.loc[df_tot['shift'] == 0]

x = df_tot.val_e.values
y = df_tot.val_0.values


plt.plot(correlate(x, y))


lag = np.argmax(correlate(x, y))
lag






