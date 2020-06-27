# Machine Unlearning with Sisa
### Lucas Bourtoule, Varun Chandrasekaran, Christopher Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, Nicolas Papernot
This repository contains the core code used in the SISA experiments of our [Machine Unlearning](https://arxiv.org/abs/1912.03817) paper along with some example scripts.

You can start runing experiments by having a look at the readme in the purchase example folder at ``example-scripts/purchase-sharding``.

``sisa.py`` is the script that trains a given shard. It should be run as many times as the number of shards.

## Citing this work

If you use this repository for academic research, you are highly encouraged
(though not required) to cite our paper:

```
@inproceedings{bourtoule2021machine,
  title={Machine Unlearning},
  author={Lucas Bourtoule and Varun Chandrasekaran and Christopher Choquette-Choo and Hengrui Jia and Adelin Travers and Baiwu Zhang and David Lie and Nicolas Papernot},
  booktitle={Proceedings of the 42nd IEEE Symposium on Security and Privacy},
  year={2021}
}
```
