Purchase dataset could be obtained [here](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/datasets). 
To customize number of labels, save the unlabelled data at `machine-unlearning/datasets/purchase/data.npy` and run `prepare_data.py`.

The following scripts allow to run a sharding experiment on purchase dataset.

1- Create a container with a specified number of shards:
init.sh 5

2- Train the shards in the container:
train.sh 5

3- Compute shard predictions:
predict.sh 5

4- Retrieve experimental data as a CSV:
data.sh 5
