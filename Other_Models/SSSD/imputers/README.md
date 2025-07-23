

## How to use the models
## $SSSD^{S4}$ and $SSSD^{SA}$ 
can be accesed through the command line, use their respective configuration files from the config/ directory. Load and reshape the datasets into the train.py and inference.py files accordly. 


## $CSDI^{S4}$ 
can be accesed as a python module in a notebook, with three main attributions:
```
from CSDIS4 import CSDIS4Imputer
imputer = CSDIS4Imputer()
imputer.train(data, masking, missing_ratio, batch_size) # for training
imputer.load_weights('path_to_model', 'path_to_config') # after training
imputations = imputer.impute(data, mask, number_of_samples) # sampling
```

## Fast experiment - Mujoco dataset 90% random missing
```
python3 train.py -c config/config_SSSDS4.json
python3 inference.py -c config/config_SSSDS4.json
```
