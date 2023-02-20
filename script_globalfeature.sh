#!/bin/sh
screen -d -m -r -S Transi  python model.py -i utils/hyperparameters/ER/er_params_Transitivity.json 
screen -d -m -r  -S Resilience  python model.py -i utils/hyperparameters/ER/er_params_Resilience.json 
screen -d -m -r  -S None python model.py -i utils/hyperparameters/ER/er_params_None.json    
screen -d -m -r  -S All  python model.py -i utils/hyperparameters/ER/er_params_All.json    
screen -d -m -r  -S Density  python model.py -i utils/hyperparameters/ER/er_params_Density.json 
screen -d -m -r  -S Hetero  python model.py -i utils/hyperparameters/ER/er_params_Heterogeneity.json   
screen -d -m -r  -S Gini  python model.py -i utils/hyperparameters/ER/er_params_Gini.json  
screen -d -m -r  -S Entropy  python model.py -i utils/hyperparameters/ER/er_params_Entropy.json