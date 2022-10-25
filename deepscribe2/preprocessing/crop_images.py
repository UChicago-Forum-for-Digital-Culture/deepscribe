# take a raw JSON export from OCHRE
# and use the dimensions of the hotspots to remove backdrop from the images.

import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict

import cv2
from joblib import Parallel, delayed
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--json", help="Raw detectron2-formatted JSON from ochre ")
    parser.add_argument("--raw_imgs", help="Raw image directory")
    parser.add_argument(
        "--cropped_imgs",
        help="Output for cropped images. If this directory doesn't exist, it will be created.",
    )
    parser.add_argument("--cropped_json", help="output for cropped JSON.")

    return parser.parse_args()


def crop_image(
    img_data: Dict, imgfolder: str, outfolder: str, error_on_missing: bool = True
) -> Dict:
    new_data = deepcopy(img_data)

    points = [anno["bbox"] for anno in img_data["annotations"]]

    # dealing with negative-valued coordinates
    min_x0 = int(max(0, min([pt[0] for pt in points])))
    max_x1 = int(max([pt[2] for pt in points]))

    min_y0 = int(max(0, min([pt[1] for pt in points])))
    max_y1 = int(max([pt[3] for pt in points]))

    img = cv2.imread(f"{imgfolder}/{img_data['file_name']}")

    if img is None:
        print(
            img_data["file_name"]
            + " does not appear to exist or is not properly formatted."
        )
        if not error_on_missing:
            return None
        else:
            raise RuntimeError()

    cropped = img[min_y0:max_y1, min_x0:max_x1, :]

    # adjust points - new origin is min_x0, min_y0
    new_data["height"] = cropped.shape[0]
    new_data["width"] = cropped.shape[1]

    for anno in new_data["annotations"]:
        old_bbox = anno["bbox"]
        anno["bbox"] = [
            max(0, old_bbox[0]) - min_x0,
            old_bbox[1] - min_y0,
            old_bbox[2] - min_x0,
            old_bbox[3] - min_y0,
        ]

    cv2.imwrite(f"{outfolder}/{new_data['file_name']}", cropped)

    return new_data


def main():
    args = parse_args()

    with open(args.json, "r") as infile:
        raw_dset = json.load(infile)

    print(f"{len(raw_dset)} entries in the dataset loaded from {args.json}.")

    os.mkdir(args.cropped_imgs)

    cropped_entries = Parallel(n_jobs=10)(
        delayed(crop_image)(
            entry, args.raw_imgs, args.cropped_imgs, error_on_missing=True
        )
        for entry in tqdm(raw_dset)
    )

    with open(args.cropped_json, "w") as outf:
        json.dump(cropped_entries, outf)


if __name__ == "__main__":
    main()
