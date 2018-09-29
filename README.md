# PROFILE_READER
**Modules for database creation from various data sources**

## Instructions
* git clone git@github.com:mcsoini/PROFILE_READER.git
* create config_local.py file which sets the absolute data directory path BASE_DIR. Otherwise the config.py file will remind you.
* create a profiles_raw postgresql schema
* an existing lp_input schema is required to instantiate the Maps class
* run the whole call_all.py script

Note: makes use of some grimsel (github.com/mcsoini/grimsel) auxiliary modules
