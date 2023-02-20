# produce raw hotspot images for classification task.
# converts a dataset file into a a dataset amenable to the
# torchvision ImageFolder format.
import os
from torchvision.io import read_image, write_jpeg
from tqdm import tqdm
from argparse import ArgumentParser
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--json", help="Raw detectron2-formatted JSON from ochre ")
    parser.add_argument("--raw_imgs", help="Raw image directory")
    parser.add_argument(
        "--split_imgs",
        help="Output for split images. If this directory doesn't exist, it will be created.",
    )

    return parser.parse_args()


def split_entry(entry, img_base, output_base):
    # iterates through every annotation, chops out the hotspot, saves to disk.

    img = read_image(f"{img_base}/{entry['file_name']}")

    for i, annotation in enumerate(entry["annotations"]):
        x0, y0, x1, y1 = annotation["bbox"]
        hotspot = img[:, y0:y1, x0:x1]
        # if the class subdir doesn't exist, create it
        subdir = f"{output_base}/{annotation['category_id']}"
        os.makedirs(subdir, exist_ok=True)
        write_jpeg(
            hotspot, f"{subdir}/{annotation['category_id']}_{i}_{entry['file_name']}"
        )


def main():
    args = parse_args()

    with open(args.json) as inf:
        dataset = json.load(inf)

    for entry in tqdm(dataset):
        split_entry(entry, args.raw_imgs, args.split_imgs)


if __name__ == "__main__":
    main()
