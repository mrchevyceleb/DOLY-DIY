"""Install the pinned, checksum-verified local person detector."""
import hashlib
from pathlib import Path
import shutil
import urllib.request

URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
       "47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_nanodet/"
       "object_detection_nanodet_2022nov.onnx")
SHA256 = "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"


def install(path="/opt/spark/models/nanodet.onnx"):
    import cv2  # supplied by the Doly SDK image; do not replace its native build
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    license_path = Path(__file__).resolve().parent.parent / "licenses/NanoDet-APACHE-2.0.txt"
    shutil.copy2(license_path, target.parent / "NanoDet-APACHE-2.0.txt")
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        return
    with urllib.request.urlopen(URL, timeout=30) as response:
        data = response.read(8*1024*1024)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("Person detection model checksum mismatch")
    temporary = target.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(target)


if __name__ == "__main__":
    install()
