#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mcsoini
"""

import grimsel.auxiliary.aux_sql_func as aql


def build_timestamp_template(db, sc, yr_min, yr_max):
    
    '''
    Parameters:
    yr_min/yr_max -- first year as int or string
    db -- database
    '''
    
    # get time map
    exec_str = (
    '''
    DROP TABLE IF EXISTS {sc_agg}.timestamp_template CASCADE;
    
    WITH yr_rg AS (
        SELECT {yr_min}::VARCHAR AS yr_min,
               {yr_max}::VARCHAR AS yr_max
    ), mt_map AS (
        SELECT mt_id::SMALLINT, season_id::SMALLINT
        FROM (VALUES (11, 0), (0, 0), (1, 0), (2, 1), (3, 1), (4, 1),
                     (5, 2), (6, 2), (7, 2), (8, 3), (9, 3), (10, 3))
        AS temp (mt_id, season_id)
    ), mt_map2 AS (
        SELECT mt_id::SMALLINT, season2_id::SMALLINT
        FROM (VALUES (11, 0), (0, 0), (1, 0), (2, 0), (3, 1), (4, 1),
                     (5, 1), (6, 1), (7, 1), (8, 1), (9, 0), (10, 0))
        AS temp (mt_id, season2_id)
    ), season_name_map AS (
        SELECT season_id::SMALLINT, season::VARCHAR
        FROM (VALUES (0, 'winter'), (1, 'spring'), (2, 'summer'), (3, 'fall'))
        AS temp (season_id, season)
    ), season2_name_map AS (
        SELECT season2_id::SMALLINT, season2::VARCHAR
        FROM (VALUES (0, 'winter'), (1, 'summer'))
        AS temp (season2_id, season2)
    ), hr_map AS (
        SELECT hour::SMALLINT, daytime_name::VARCHAR(10)
        FROM (VALUES (0, 'Night'), (1, 'Night'), (2, 'Night'), (3, 'Night'), (4, 'Night'),
                     (5, 'Morning'), (6, 'Morning'), (7, 'Morning'), (8, 'Morning'), (9, 'Morning'),
                     (10, 'Noon'), (11, 'Noon'), (12, 'Noon'), (13, 'Noon'), (14, 'Noon'),
                     (15, 'Afternoon'), (16, 'Afternoon'), (17, 'Afternoon'), (18, 'Afternoon'),
                     (19, 'Evening'), (20, 'Evening'), (21, 'Evening'), (22, 'Evening'), (23, 'Evening')
                      )
        AS temp (hour, daytime_name)
    ), hr_map_min AS ( -- initial hour of that daytime
        SELECT daytime_name, MIN(hour) AS daytime_min
        FROM hr_map
        GROUP BY daytime_name
    ), wd_map AS (
        SELECT dow::SMALLINT, is_weekend::SMALLINT
        FROM (VALUES (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 1), (7, 1))
        AS temp (dow, is_weekend)
    ), wd_name_map AS (
        SELECT is_weekend::SMALLINT, is_weekend_name::VARCHAR
        FROM (VALUES (0, 'weekday'), (1, 'weekend'))
        AS temp (is_weekend, is_weekend_name)
    ), tb_dt_base AS (
        SELECT datetime,
            ((ROW_NUMBER() OVER (ORDER BY datetime) - 1) % 8760)::SMALLINT AS slot,
            extract(year FROM datetime)::SMALLINT AS year,
            ('yr' || extract(YEAR FROM datetime))::CHAR(6) AS swyr_vl,
            (extract(ISODOW FROM datetime))::SMALLINT AS dow,
            (extract(DOY FROM datetime))::SMALLINT AS doy,
            (extract(HOUR FROM datetime))::SMALLINT AS hour,
            (extract(WEEK FROM datetime))::SMALLINT - 1 AS wk_id,
            (extract(DAY FROM datetime))::SMALLINT AS dom,
            (extract(MONTH FROM datetime) - 1)::SMALLINT AS mt_id,
            to_char(datetime, 'W')::SMALLINT - 1 AS wom
        FROM (SELECT GENERATE_SERIES(((SELECT yr_min FROM yr_rg) || '-01-01T00:00:00Z')::timestamp,
                                     ((SELECT yr_max FROM yr_rg) || '-12-31T23:59:59Z')::timestamp,'1 hour') AS datetime) AS dt
        WHERE NOT ((DATE_PART('month', datetime) = 2) AND (DATE_PART('day', datetime) = 29))
    )
    SELECT *,
        ((dow - 1) * 24 + hour)::SMALLINT AS how,
        ((dom - 1) * 24 + hour)::SMALLINT AS hom,
        LPAD(daytime_min::VARCHAR, 2, '0') || '_' || daytime_name AS daytime
    INTO {sc_agg}.timestamp_template
    FROM tb_dt_base
    NATURAL LEFT JOIN mt_map
    NATURAL LEFT JOIN mt_map2
    NATURAL LEFT JOIN season_name_map
    NATURAL LEFT JOIN season2_name_map
    NATURAL LEFT JOIN wd_map
    NATURAL LEFT JOIN wd_name_map
    NATURAL LEFT JOIN hr_map
    NATURAL LEFT JOIN hr_map_min;
    
    -- ADDITIONAL DERIVED COLUMNS
    
    ALTER TABLE {sc_agg}.timestamp_template
    ADD COLUMN IF NOT EXISTS hom_w SMALLINT;
    UPDATE {sc_agg}.timestamp_template
    SET hom_w = how + 168 * wom;
    
    DROP TABLE IF EXISTS {sc_agg}.daytime_map CASCADE;
    WITH dn_map AS (
        SELECT hour::SMALLINT, daytime_dn::VARCHAR(10)
        FROM (VALUES (0, 'Night'), (1, 'Night'), (2, 'Night'), (3, 'Night'), (4, 'Night'),
                     (5, 'Night'), (6, 'Night'), (7, 'Night'), (8, 'Day'), (9, 'Day'),
                     (10, 'Day'), (11, 'Day'), (12, 'Day'), (13, 'Day'), (14, 'Day'),
                     (15, 'Day'), (16, 'Day'), (17, 'Day'), (18, 'Day'),
                     (19, 'Night'), (20, 'Night'), (21, 'Night'), (22, 'Night'), (23, 'Night')
                      )
        AS temp (hour, daytime_dn)
    )
    SELECT DISTINCT hour, daytime, daytime_dn
    INTO {sc_agg}.daytime_map
    FROM {sc_agg}.timestamp_template
    NATURAL LEFT JOIN dn_map;
    '''.format(sc_agg=sc, yr_min=yr_min, yr_max=yr_max)
    )
    aql.exec_sql(exec_str, db=db)
