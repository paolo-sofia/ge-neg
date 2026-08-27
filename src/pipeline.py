import pathlib

from ge_neg.border_identifier import BorderIdentifier
from ge_neg.contrast_booster import ContrastBoosterGenetic
from ge_neg.inversion import inversion_and_density_balance
from ge_neg.preprocessing import (
    expose_to_the_right,
    load_and_normalize,
    remove_scanner_light,
)
from ge_neg.utils import apply_s_curve, save_to_file
from ge_neg.white_balance import film_base_wb, scene_wb


def process_image(
    input_path: pathlib.Path, output_path: pathlib.Path, debug: bool = False
) -> bool:
    print("[MODULO 0] - Caricamento e normalizzazione immagine")
    img = load_and_normalize(input_path)

    print("[MODULO 0] - Rimozione luce scanner")
    trimmed_image = remove_scanner_light(img)
    if debug:
        _ = save_to_file(trimmed_image, output_path, "scanner_crop")

    print("[MODULO 0] - Compensazione dell'esposizione con ETTR")
    ettr = expose_to_the_right(trimmed_image)
    if debug:
        _ = save_to_file(ettr, output_path, "exposure_compensation")

    print("[MODULO 1] - Identify borders and compute film base")
    border_identifier = BorderIdentifier(img=ettr)
    border_identifier.find_borders()
    photo_pixels = border_identifier.get_image()

    if debug:
        _ = save_to_file(photo_pixels, output_path, "tagliata")

    print("[MODULO 2] - Film base white balance")
    image_wb = film_base_wb(
        photo_pixels, film_base_color=border_identifier.get_film_base()
    )
    if debug:
        _ = save_to_file(image_wb, output_path, "wb")

    print("[MODULO 3] - Inversion and density balance")
    positive_img = inversion_and_density_balance(image_wb)

    if debug:
        _ = save_to_file(positive_img, output_path, "positive")

    print("[MODULO 3] - Scene white balance")
    img_scene_wb = scene_wb(positive_img)

    if debug:
        _ = save_to_file(img_scene_wb, output_path, "wb_scena")

    print("[MODULO 4] - Contrast booster")
    contrast_booster = ContrastBoosterGenetic(img_scene_wb)
    contrast_booster.run()
    x0, k, h = contrast_booster.genetic_optimizer.best_solution()[0]
    print(
        f"[MODULO 4] - Parametri migliori per curva di contrasto: x0 = {x0} - k = {k} - h = {h}"
    )
    contrasted_image = apply_s_curve(
        img_scene_wb, *contrast_booster.genetic_optimizer.best_solution()[0]
    )
    return save_to_file(
        contrasted_image, output_path, "contrast_booster" if debug else ""
    )
