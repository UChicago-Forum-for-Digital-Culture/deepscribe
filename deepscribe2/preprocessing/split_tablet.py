# split a dataset by tablet instead of by image.

from argparse import ArgumentParser
import numpy as np
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--json", help="input OCHRE exported JSON file")
    parser.add_argument(
        "--splits", nargs="+", help="Splits. Must add up to 1.", type=float
    )
    parser.add_argument("--prefix", help="Output file prefix")
    parser.add_argument(
        "--fold_suffixes",
        help="Suffixes to use for folds. If not provided, will be assigned numerically.",
        nargs="+",
        required=False,
    )
    return parser.parse_args()


def split_by_tablet(dset, splits, tablet_id_key="text_id", seed=0):
    # split data by text into disjoint folds
    all_texts = [entry[tablet_id_key] for entry in dset]

    unique_texts = np.unique(all_texts)

    print(f"{len(unique_texts)} unique texts in dataset")
    np.random.default_rng(seed).shuffle(unique_texts)

    splits = np.array(splits)

    if not sum(splits) == 1:
        raise ValueError(f"{splits} does not add up to 1.")

    split_inds = (np.cumsum(splits[:-1]) * len(unique_texts)).astype(int)

    folds = []

    for fold_texts in np.split(unique_texts, split_inds):
        fold_data = [entry for entry in dset if entry[tablet_id_key] in fold_texts]
        folds.append(fold_data)

    return folds


def main():
    args = parse_args()

    with open(args.json) as inf:
        dset = json.load(inf)

    folds = split_by_tablet(dset, args.splits)

    if args.fold_suffixes and not len(args.fold_suffixes) == len(args.splits):
        raise ValueError(f"{args.fold_suffixes} does not match {args.split}")

    fold_suffixes = (
        args.fold_suffixes if args.fold_suffixes else range(len(args.splits))
    )

    for fold, suffix in zip(folds, fold_suffixes):
        with open(f"{args.prefix}_{suffix}.json", "w") as outf:
            json.dump(fold, outf)


if __name__ == "__main__":
    main()
