# Instruments_cl_ori

Original PESO Instruments data and SID mapping. Existing SID collisions are retained.

The five temporal block JSON files include block 0. Item metadata are ordinary
JSON dictionaries split into `.item.part001.json`, `.item.part002.json`, etc.;
the data loader reads these directly. No model checkpoints are included.
