These scripts allow to run a sharding experiment on purchase dataset.

1- Create a container with a specified number of shards:
init.sh 5

2- Train the shards in the container:
train.sh 5

3- Compute shard predictions:
predict.sh 5

4- Retrieve experimental data as a CSV:
data.sh 5
