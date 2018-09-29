import os

from xlrd import open_workbook
import pandas as pd

from grimsel_h.auxiliary.aux_general import print_full, expand_rows, read_xlsx_table


def get_chp_profiles(target_dir=None):

    print('Directory get_chp_profiles:', os.getcwd())

    fn = 'heat_demand_profiles_20170614.xlsx'
    dct = '../../../SHARED_DATA/HEAT_DEMAND_PROFILE/'
    wb = open_workbook(dct + fn)
    profile_heat_cols = range(31)
    df_profile_heat = read_xlsx_table(wb, ['TOT'], columns=profile_heat_cols)

    years = df_profile_heat.loc[1]
    ctrys = df_profile_heat.loc[0]
    cols =  [(years[i], ctrys[i]) for i in range(len(years))]

    df_profile_heat.columns = ['hoy'] + cols[1:]
    df_profile_heat = df_profile_heat.drop([0,1], axis=0).reset_index(drop=True)
    df_profile_heat = df_profile_heat.set_index('hoy')
    df_profile_heat.columns = pd.MultiIndex.from_tuples(df_profile_heat.columns,
                                                        names=['year','country'])

    df_profile_heat = df_profile_heat.stack(['year', 'country']).reset_index()

    df_profile_heat = df_profile_heat.loc[df_profile_heat.year == 2015]

    df_profile_heat.columns = ([c for c in df_profile_heat.columns[:-1]]
                             + ['power'])
    df_profile_heat['power'] = df_profile_heat['power'].apply(float)

    # scale to 50% of the maximum

    df_profile_heat['power'] = df_profile_heat.groupby(['year', 'country'])['power'].transform(lambda x: x/x.max() * 0.5)

    df_profile_heat = df_profile_heat.loc[df_profile_heat.year == 2015]

    df_profile_heat['node'] = df_profile_heat['country'] + '0'

    df_profile_heat = df_profile_heat[['hoy', 'node', 'power']]

    if target_dir == None:
        target_dir = dct

    df_profile_heat.to_csv(target_dir + 'heat_profile.csv', index=False)