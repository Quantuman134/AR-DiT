"""Generate the four tiny PNG fixtures used by ``tests/test_dataset.py``.

Layout produced (relative to this file's directory)::

    class_00/img.png    # 32x32 solid red
    class_01/img.png    # 32x32 solid green
    class_02/img.png    # 32x32 solid blue
    class_03/img.png    # 32x32 solid black

The four PNGs are checked into the repository so that the CIFAR-10
dataset adapter (:class:`data.cifar10.CIFAR10Dataset`) can be exercised
end-to-end on real encoded image bytes without needing the actual
CIFAR-10 archive.

This script exists so that anyone can regenerate the fixtures
identically:

    python tests/fixtures/images/make_fixtures.py

The four class-directory names are chosen so that ``ImageFolder``'s
alphabetical sort assigns labels ``0..3`` in the visually-obvious order
red → green → blue → black.  Distinct colours also make it easy to
eyeball the fixtures in an image viewer if a test ever fails.

See doc/Test.md §"tests/test_dataset.py" for the tests these fixtures
support.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# (subdir_name, RGB colour tuple) — order here is *not* what determines the
# integer label.  ImageFolder sorts subdir names alphabetically, and the
# names below happen to already be sorted (class_00 < class_01 < ...) so
# label i corresponds to CLASSES[i].
CLASSES: list[tuple[str, tuple[int, int, int]]] = [
    ("class_00", (255, 0, 0)),      # red   -> label 0
    ("class_01", (0, 255, 0)),      # green -> label 1
    ("class_02", (0, 0, 255)),      # blue  -> label 2
    ("class_03", (0, 0, 0)),        # black -> label 3
]

IMG_SIZE = (32, 32)


def main() -> None:
    here = Path(__file__).resolve().parent
    for subdir, rgb in CLASSES:
        out_dir = here / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", IMG_SIZE, color=rgb)
        img.save(out_dir / "img.png", format="PNG", optimize=True)
        print(f"wrote {out_dir / 'img.png'}  ({rgb})")


if __name__ == "__main__":
    main()
