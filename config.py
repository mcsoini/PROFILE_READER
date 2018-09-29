#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import PROFILE_READER.config_local as conf_local
    BASE_DIR = conf_local.BASE_DIR
except:
    raise RuntimeError('Please set BASE_DIR variable in '
                       'PROFILE_READER/config_local.py, e.g. '
                       'BASE_DIR = os.path.abspath(\'../../DATA/\')')



