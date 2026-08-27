import argparse
import pathlib
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

    return list(set(input_images).difference(output_images))


def main(
    image_type_to_process: str = "", image_description_to_process: str = ""
) -> None:
    input_path = pathlib.Path("/home/paolo/git/ge-neg/tests/input_images/")
    output_folder = pathlib.Path("/home/paolo/git/ge-neg/tests/output_images/")

    paths = input_path.rglob("**")
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        image_type: str = path.parents[0].stem
        image_description: str = path.stem

        if image_type_to_process and image_type != image_type_to_process:
            continue

        if (
            image_description_to_process
            and image_description != image_description_to_process
        ):
            continue

        output_path = output_folder / image_type / image_description
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            print(
                f" ============================== Processing image {image_type} - {image_description} =============================="
            )
            start_time = time()
            is_image_saved = process_image(path, output_path)
            elapsed = time() - start_time

            print(
                f"Image {path.stem} processed in {int(elapsed // 60)}:{int(elapsed % 60)} minutes"
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ge-neg",
        description="Processa automaticamente i negativi",
    )
    _ = parser.add_argument("-t", "--type", type=str)
    _ = parser.add_argument("-d", "--description", type=str)

    args: argparse.Namespace = parser.parse_args()
    main(args.type, args.description)
