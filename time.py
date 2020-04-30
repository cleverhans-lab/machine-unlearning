import pandas as pd

import argparse

# Compute stats based on the execution time (cumulated feed-forward + backprop.) of the shards

parser = argparse.ArgumentParser()
parser.add_argument('--container', help="Name of the container")
args = parser.parse_args()

t = pd.read_csv('containers/{}/times/times.tmp'.format(args.container), names=['time'])
print('{},{}'.format(t['time'].sum(),t['time'].mean()))