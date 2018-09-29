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
from grimsel.auxiliary.aux_general import read_xlsx_table
from xlrd import open_workbook

import PROFILE_READER.profile_reader as profile_reader

reload(profile_reader)

class MonthlyProductionReader(profile_reader.ProfileReader):
    '''
    '''

    dict_sql_default = dict(sc='profiles_raw', tb='monthly_production')
    data_dir = os.path.normpath('VARIOUS_SOURCES_MONTHLY_SUMS')

    tb_cols = [('fl', 'VARCHAR'),
               ('mt_id', 'SMALLINT'),
               ('nd', 'VARCHAR'),
               ('run_id', 'SMALLINT'),
               ('erg', 'DOUBLE PRECISION'),
               ('input', 'VARCHAR')]
    tb_pk = ['fl', 'mt_id', 'nd', 'input', 'run_id']

    exclude_substrings=[]


    def __init__(self, kw_dict):
        super().__init__(**kw_dict)

        self.get_fn_list()


    def read(self, fn):

        wb = open_workbook(fn)
        df_add = read_xlsx_table(wb, sheets=['Tabelle1'], columns=['fl', 'mt_id', 'nd', 'run_id', 'erg', 'input'])

        return df_add

    def read_all(self):
        
        fn = self.fn_list[0]
        self.df_tot = self.read(fn)

    
        self.append_to_sql(self.df_tot)

if __name__ == '__main__':

    kw_dict = dict(dict_sql=dict(db='storage2'),
                   exclude_substrings=[],
                   tm_filt={'year': range(2005, 2018)},
                   ext=['xlsx'])


    op = MonthlyProductionReader(kw_dict)
    self = op
    fn = self.fn_list[0]
    op.read_all()


# %%

