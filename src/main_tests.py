import argparse
import pathlib

from src.ge_neg.config_loader import db_path, valid_extensions
from src.pipeline import ImageProcessor


def find_missing_images(
    input_path: pathlib.Path, output_path: pathlib.Path
) -> list[str]:
    input_images: list[str] = [
        x.name for x in input_path.rglob("**") if x.suffix in valid_extensions
    ]
    output_images: list[str] = [
        x.name for x in output_path.rglob("**") if x.suffix in valid_extensions
    ]

    return list(set(input_images).difference(output_images))


def main(
    image_type_to_process: str = "", image_description_to_process: str = ""
) -> None:
    input_path = pathlib.Path("/home/paolo/git/ge-neg/tests/input_images/")
    output_folder = pathlib.Path("/home/paolo/git/ge-neg/tests/output_images/")

    paths = input_path.rglob("**")
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in valid_extensions:
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

        print(
            f" ============================== Processing image {image_type} - {image_description} =============================="
        )
        image_processor: ImageProcessor = ImageProcessor(path, output_path)
        image_processor.run(db_path)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ge-neg",
        description="Processa automaticamente i negativi",
    )
    _ = parser.add_argument("-t", "--type", type=str)
    _ = parser.add_argument("-d", "--description", type=str)

    args: argparse.Namespace = parser.parse_args()
    # main(args.type, args.description)

    path = pathlib.Path(
        "/home/paolo/Immagini/analog_images/bronze/nikon_coolscan/test/img_060_00.tif"
    )
    output_path = pathlib.Path(
        "/home/paolo/Immagini/analog_images/silver/nikon_coolscan/test/img_060_00.tif"
    )
    image_processor: ImageProcessor = ImageProcessor([], path, output_path)
    image_processor.run(db_path)
