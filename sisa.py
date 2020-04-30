import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn.functional import one_hot
from sharded import sizeOfShard, getShardHash, fetchShardBatch, fetchTestBatch
import os
from glob import glob
from time import time
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="purchase", help="Architecture to use, default purchase"
)

parser.add_argument(
    "--train", action="store_true", help="Perform SISA training on the shard"
)
parser.add_argument("--test", action="store_true", help="Compute shard predictions")

parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Train for the specified number of epochs, default 20",
)
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Size of the batches, relevant for both train and test, default 16",
)
parser.add_argument(
    "--dropout_rate",
    default=0.4,
    type=float,
    help="Dropout rate, if relevant, default 0.4",
)
parser.add_argument(
    "--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001"
)

parser.add_argument("--optimizer", default="sgd", help="Optimizer, default sgd")

parser.add_argument(
    "--output_type",
    default="argmax",
    help="Type of outputs to be used in aggregation, can be either argmax or softmax, default argmax",
)

parser.add_argument("--container", help="Name of the container")
parser.add_argument("--shard", type=int, help="Index of the shard to train/test")
parser.add_argument(
    "--slices", default=1, type=int, help="Number of slices to use, default 1"
)
parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Location of the datasetfile, default datasets/purchase/datasetfile",
)

parser.add_argument(
    "--chkpt_interval",
    default=1,
    type=int,
    help="Interval (in epochs) between two chkpts, -1 to disable chackpointing, default 1",
)
parser.add_argument(
    "--label",
    default="latest",
    help="Label to be used on simlinks and outputs, default latest",
)
args = parser.parse_args()

# Import the architecture.
from importlib import import_module

model_lib = import_module("architectures.{}".format(args.model))

# Retrive dataset metadata.
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())
input_shape = tuple(datasetfile["input_shape"])
nb_classes = datasetfile["nb_classes"]

# Use GPU if available.
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # pylint: disable=no-member

# Instantiate model and send to selected device.
model = model_lib.Model(input_shape, nb_classes, dropout_rate=args.dropout_rate)
model.to(device)

# Instantiate loss and optimizer.
loss_fn = CrossEntropyLoss()
if args.optimizer == "adam":
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
else:
    raise "Unsupported optimizer"

if args.train:
    shard_size = sizeOfShard(args.container, args.shard)
    slice_size = shard_size // args.slices
    avg_epochs_per_slice = (
        2 * args.slices / (args.slices + 1) * args.epochs / args.slices
    )
    loaded = False

    for sl in range(args.slices):
        # Get slice hash using sharded lib.
        slice_hash = getShardHash(
            args.container, args.label, args.shard, until=(sl + 1) * slice_size
        )

        # If checkpoints exists, skip the slice.
        if not os.path.exists(
            "containers/{}/cache/{}.pt".format(args.container, slice_hash)
        ):
            # Initialize state.
            elapsed_time = 0
            start_epoch = 0
            slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(
                sl * avg_epochs_per_slice
            )

            # If weights are already in memory (from previous slice), skip loading.
            if not loaded:
                # Look for a recovery checkpoint for the slice.
                recovery_list = glob(
                    "containers/{}/cache/{}_*.pt".format(args.container, slice_hash)
                )
                if len(recovery_list) > 0:
                    print(
                        "Recovery mode for shard {} on slice {}".format(args.shard, sl)
                    )

                    # Load weights.
                    model.load_state_dict(torch.load(recovery_list[0]))
                    start_epoch = int(
                        recovery_list[0].split("/")[-1].split(".")[0].split("_")[1]
                    )

                    # Load time
                    with open(
                        "containers/{}/times/{}_{}.time".format(
                            args.container, slice_hash, start_epoch
                        ),
                        "r",
                    ) as f:
                        elapsed_time = float(f.read())

                # If there is no recovery checkpoint and this slice is not the first, load previous slice.
                elif sl > 0:
                    previous_slice_hash = getShardHash(
                        args.container, args.label, args.shard, until=sl * slice_size
                    )

                    # Load weights.
                    model.load_state_dict(
                        torch.load(
                            "containers/{}/cache/{}.pt".format(
                                args.container, previous_slice_hash
                            )
                        )
                    )

                # Mark model as loaded for next slices.
                loaded = True

            # If this is the first slice, no need to load anything.
            elif sl == 0:
                loaded = True

            # Actual training.
            train_time = 0.0

            for epoch in range(start_epoch, slice_epochs):
                epoch_start_time = time()

                for images, labels in fetchShardBatch(
                    args.container,
                    args.label,
                    args.shard,
                    args.batch_size,
                    args.dataset,
                    until=(sl + 1) * slice_size if sl < args.slices - 1 else None,
                ):

                    # Convert data to torch format and send to selected device.
                    gpu_images = torch.from_numpy(images).to(
                        device
                    )  # pylint: disable=no-member
                    gpu_labels = torch.from_numpy(labels).to(
                        device
                    )  # pylint: disable=no-member

                    forward_start_time = time()

                    # Perform basic training step.
                    logits = model(gpu_images)
                    loss = loss_fn(logits, gpu_labels)

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    train_time += time() - forward_start_time

                # Create a checkpoint every chkpt_interval.
                if (
                    args.chkpt_interval != -1
                    and epoch % args.chkpt_interval == args.chkpt_interval - 1
                ):
                    # Save weights
                    torch.save(
                        model.state_dict(),
                        "containers/{}/cache/{}_{}.pt".format(
                            args.container, slice_hash, epoch
                        ),
                    )

                    # Save time
                    with open(
                        "containers/{}/times/{}_{}.time".format(
                            args.container, slice_hash, epoch
                        ),
                        "w",
                    ) as f:
                        f.write("{}\n".format(train_time + elapsed_time))

                    # Remove previous checkpoint.
                    if os.path.exists(
                        "containers/{}/cache/{}_{}.pt".format(
                            args.container, slice_hash, epoch - args.chkpt_interval
                        )
                    ):
                        os.remove(
                            "containers/{}/cache/{}_{}.pt".format(
                                args.container, slice_hash, epoch - args.chkpt_interval
                            )
                        )
                    if os.path.exists(
                        "containers/{}/times/{}_{}.time".format(
                            args.container, slice_hash, epoch - args.chkpt_interval
                        )
                    ):
                        os.remove(
                            "containers/{}/times/{}_{}.time".format(
                                args.container, slice_hash, epoch - args.chkpt_interval
                            )
                        )

            # When training is complete, save slice.
            torch.save(
                model.state_dict(),
                "containers/{}/cache/{}.pt".format(args.container, slice_hash),
            )
            with open(
                "containers/{}/times/{}.time".format(args.container, slice_hash), "w"
            ) as f:
                f.write("{}\n".format(train_time + elapsed_time))

            # Remove previous checkpoint.
            if os.path.exists(
                "containers/{}/cache/{}_{}.pt".format(
                    args.container, slice_hash, args.epochs - args.chkpt_interval
                )
            ):
                os.remove(
                    "containers/{}/cache/{}_{}.pt".format(
                        args.container, slice_hash, args.epochs - args.chkpt_interval
                    )
                )
            if os.path.exists(
                "containers/{}/times/{}_{}.time".format(
                    args.container, slice_hash, args.epochs - args.chkpt_interval
                )
            ):
                os.remove(
                    "containers/{}/times/{}_{}.time".format(
                        args.container, slice_hash, args.epochs - args.chkpt_interval
                    )
                )

            # If this is the last slice, create a symlink attached to it.
            if sl == args.slices - 1:
                os.symlink(
                    "{}.pt".format(slice_hash),
                    "containers/{}/cache/shard-{}:{}.pt".format(
                        args.container, args.shard, args.label
                    ),
                )
                os.symlink(
                    "{}.time".format(slice_hash),
                    "containers/{}/times/shard-{}:{}.time".format(
                        args.container, args.shard, args.label
                    ),
                )

        elif sl == args.slices - 1:
            os.symlink(
                "{}.pt".format(slice_hash),
                "containers/{}/cache/shard-{}:{}.pt".format(
                    args.container, args.shard, args.label
                ),
            )
            if not os.path.exists(
                "containers/{}/times/shard-{}:{}.time".format(
                    args.container, args.shard, args.label
                )
            ):
                os.symlink(
                    "null.time",
                    "containers/{}/times/shard-{}:{}.time".format(
                        args.container, args.shard, args.label
                    ),
                )


if args.test:
    # Load model weights from shard checkpoint (last slice).
    model.load_state_dict(
        torch.load(
            "containers/{}/cache/shard-{}:{}.pt".format(
                args.container, args.shard, args.label
            )
        )
    )

    # Compute predictions batch per batch.
    outputs = np.empty((0, nb_classes))
    for images, _ in fetchTestBatch(args.dataset, args.batch_size):
        # Convert data to torch format and send to selected device.
        gpu_images = torch.from_numpy(images).to(device)  # pylint: disable=no-member

        if args.output_type == "softmax":
            # Actual batch prediction.
            logits = model(gpu_images)
            predictions = softmax(logits, dim=1).to("cpu")  # Send back to cpu.

            # Convert back to numpy and concatenate with previous batches.
            outputs = np.concatenate((outputs, predictions.numpy()))

        else:
            # Actual batch prediction.
            logits = model(gpu_images)
            predictions = torch.argmax(logits, dim=1)  # pylint: disable=no-member

            # Convert to one hot, send back to cpu, convert back to numpy and concatenate with previous batches.
            out = one_hot(predictions, nb_classes).to("cpu")
            outputs = np.concatenate((outputs, out.numpy()))

    # Save outputs in numpy format.
    outputs = np.array(outputs)
    np.save(
        "containers/{}/outputs/shard-{}:{}.npy".format(
            args.container, args.shard, args.label
        ),
        outputs,
    )
