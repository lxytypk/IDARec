# Movies_and_TV_cl_ori

Author-constructed Movies and TV data. Each of 46,074 items has a unique four-token SID.

The five temporal block JSON files include block 0. Item metadata are ordinary
JSON dictionaries split into `.item.part001.json`, `.item.part002.json`, etc.;
the data loader reads these directly. No model checkpoints are included.
