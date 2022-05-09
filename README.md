# POI-TForecast

cfg/: config files
modules/: net.py: network model; train.py: trainer file (directly used by cmd_train.py)
cmd_train.py: train/test model in cmd
user_sim.py: if no social connections provided in the dataset, this can be used to extract social information
dataset.py: Pytorch style dataset, directly used for dataloader in train.py
dataset_social.py: used to further process the preprocessed data. The output/saved files are used in dataset.py to provide train/test data.
dataset_social_user_sim.py: same function as dataset_social.py. But this is used together with user_sim.py for dataset that does not include social connections.
