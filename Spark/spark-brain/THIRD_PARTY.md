# Person detection

`spark/person_vision.py` adapts the NanoDet preprocessing and output decoding
from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_nanodet).
Changes include person-only output, aspect-ratio padding, normalized boxes,
stable softmax, and bounds validation. The code and downloaded model are
licensed under Apache 2.0; the license is in `licenses/NanoDet-APACHE-2.0.txt`.

The original model is [NanoDet](https://github.com/RangiLyu/nanodet),
Copyright 2020 RangiLyu. OpenCV Zoo conversion and integration are by their
respective contributors. `deploy/install_vision.py` pins the model revision
and checks its SHA-256 before installation.
