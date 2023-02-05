# take a raw JSON export from OCHRE
# and use the dimensions of the hotspots to remove backdrop from the images.

import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict, Tuple

from torch import Tensor
import torch
from torchvision.io import read_image, write_jpeg
from tqdm import tqdm
from deepscribe2.utils import get_boxes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--json", help="Raw detectron2-formatted JSON from ochre ")
    parser.add_argument("--raw_imgs", help="Raw image directory")
    parser.add_argument(
        "--cropped_imgs",
        help="Output for cropped images. If this directory doesn't exist, it will be created.",
    )
    parser.add_argument(
        "--margin", help="margin in px to add around each boundary.", default=50
    )
    parser.add_argument("--cropped_json", help="output for cropped JSON.")

    return parser.parse_args()


def crop_to_box_boundaries(
    img_data: Dict,
    imgfolder: str,
    margin: int = 50,
) -> Tuple[torch.Tensor, Dict]:
    boxes = get_boxes(img_data)
    img = read_image(f"{imgfolder}/{img_data['file_name']}")
    old_height, old_width = img_data["height"], img_data["width"]
    assert (old_height, old_width) == (img.shape[1], img.shape[2])

    # dealing with negative-valued coordinates
    min_x0, min_y0 = int(max(0, boxes[:, 0].min() - margin)), int(
        max(0, boxes[:, 1].min() - margin)
    )
    max_x1, max_y1 = int(min(old_width, boxes[:, 2].max()) + margin), int(
        min(old_height, boxes[:, 3].max() + margin)
    )

    # crop image!
    cropped = img[:, min_y0:max_y1, min_x0:max_x1]

    # adjust points - new origin is min_x0, min_y0
    new_data = deepcopy(img_data)

    new_data["height"], new_data["width"] = cropped.shape[1], cropped.shape[2]

    # translate all points
    transform_matrix = torch.eye(3)
    transform_matrix[0, 2] = -min_x0
    transform_matrix[1, 2] = -min_y0
    # add projective coord
    coords0 = torch.ones((boxes.shape[0], 3))
    coords0[:, :2] = boxes[:, :2]
    coords1 = torch.ones((boxes.shape[0], 3))
    coords1[:, :2] = boxes[:, 2:]
    # remove projective coord
    transformed0 = (
        (transform_matrix @ coords0.transpose(0, 1)).to(int).transpose(0, 1)[:, :2]
    )
    transformed1 = (
        (transform_matrix @ coords1.transpose(0, 1)).to(int).transpose(0, 1)[:, :2]
    )

    for i, anno in enumerate(new_data["annotations"]):
        anno["bbox"] = transformed0[i, :].tolist() + transformed1[i, :].tolist()
    return new_data, cropped


def main():
    args = parse_args()

    with open(args.json, "r") as infile:
        raw_dset = json.load(infile)

    print(f"{len(raw_dset)} entries in the dataset loaded from {args.json}.")

    if not os.path.isdir(args.cropped_imgs):
        os.mkdir(args.cropped_imgs)

    cropped_entries = []

    for entry in tqdm(raw_dset):
        new_entry, cropped_image = crop_to_box_boundaries(
            entry, args.raw_imgs, margin=args.margin
        )
        outpath = f"{args.cropped_imgs}/{new_entry['file_name']}"
        write_jpeg(cropped_image, outpath)

    with open(args.cropped_json, "w") as outf:
        json.dump(cropped_entries, outf)


if __name__ == "__main__":
    main()
