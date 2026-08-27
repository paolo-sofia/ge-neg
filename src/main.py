import argparse
import pathlib
from threading import ExceptHookArgs
from time import time

from pipeline import process_image

VALID_EXTENSIONS = [".tiff", ".tif"]


def find_missing_images(
    input_path: pathlib.Path, output_path: pathlib.Path
) -> list[str]:
    input_images: list[str] = [
        x.name for x in input_path.rglob("**") if x.suffix in VALID_EXTENSIONS
    ]
    output_images: list[str] = [
        x.name for x in output_path.rglob("**") if x.suffix in VALID_EXTENSIONS
    ]

    return sorted(list(set(input_images).difference(output_images)))


def main(
    input_folder: pathlib.Path,
    output_folder: pathlib.Path,
    images_to_process: list[str],
) -> None:

    for image in images_to_process:
        input_path: pathlib.Path = input_folder / image
        output_path: pathlib.Path = output_folder / image
        output_folder.mkdir(parents=True, exist_ok=True)

        start_time = time()
        _ = process_image(input_path, output_path)
        elapsed = time() - start_time
        print(
            f"Image {input_path.stem} processed in {int(elapsed // 60)}:{int(elapsed % 60)} minutes"
        )

        # try:
        #     print(
        #         f" ============================== Processing image {input_path} =============================="
        #     )
        #     start_time = time()
        #     _ = process_image(input_path, output_path)
        #     elapsed = time() - start_time

        #     print(
        #         f"Image {input_path.stem} processed in {int(elapsed // 60)}:{int(elapsed % 60)} minutes"
        #     )
        # except Exception as e:
        #     print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ge-neg",
        description="Processa automaticamente i negativi",
    )
    _ = parser.add_argument("-i", "--input_path", type=pathlib.Path)
    _ = parser.add_argument("-o", "--output_path", type=pathlib.Path)
    _ = parser.add_argument("-x", "--images", type=str)
    _ = parser.add_argument("-m", "--missing", action="store_true")

    args: argparse.Namespace = parser.parse_args()

    if args.images and args.missing:
        raise Exception(
            "Impossible to set both missing and images to process, select only one"
        )

    if args.missing:
        images_to_process = find_missing_images(args.input_path, args.output_path)
        print(f"Found {len(images_to_process)} missing images to process")
        print(images_to_process)
    else:
        images_to_process = sorted(args.images.split(","))

    main(args.input_path, args.output_path, images_to_process)
