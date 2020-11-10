Before the first time running experiments on purchase dataset or to customize number of labels used, please run `prepare_data.py` at `machine-unlearning/datasets/purchase`.

The following scripts allow to run a sharding experiment on purchase dataset.

1- Create a container with a specified number of shards:
init.sh 5

2- Train the shards in the container:
train.sh 5

3- Compute shard predictions:
predict.sh 5

4- Retrieve experimental data as a CSV:
data.sh 5
