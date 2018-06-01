# coding: utf-8
import argparse
import base64
import os
import random

import cv2
import mxnet as mx
import numpy as np

from mtcnn_detector import MtcnnDetector


def get_args():
    parser = argparse.ArgumentParser(description='align faces')
    parser.add_argument('--src', help='source file')
    parser.add_argument('--dst', help='destination directory')
    parser.add_argument('--cleanlist', help='clean list')
    parser.add_argument('--workers', default=1, type=int, help='workers')
    parser.add_argument('--ngpus', default=-1, type=int, help='how many gpus to use')
    parser.add_argument('--size', default=128, type=int, help='output image is (size,size)')
    parser.add_argument('--padding', default=0.25, type=float, help='padding')
    parser.add_argument('--split', default=0, type=float, help='split ratio')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    src_file = args.src
    if not os.path.exists(src_file):
        raise ValueError("src dir not exist {}".format(src_file))

    split_ratio = args.split

    dst_dir = os.path.abspath(args.dst)

    num_gpus = args.ngpus
    if num_gpus == -1:
        num_gpus = len(mx.test_utils.list_gpus())
    if num_gpus == 0:
        ctx = mx.cpu(0)
    else:
        ctx = [mx.gpu(i) for i in range(num_gpus)]

    print("src={} dst dir={} gpu={}".format(src_file, dst_dir, num_gpus))
    s = read_clean_list(args.cleanlist)
    detector = MtcnnDetector(model_folder='model', ctx=ctx, num_worker=args.workers, accurate_landmark=True)

    file_count = 0
    with open(src_file, "r", encoding="utf-8") as f:
        last_m_id = "x"
        for line in f:
            m_id, image_search_rank, image_url, page_url, face_id, face_rectangle, face_data = line.split("\t")
            # rect = struct.unpack("ffff", base64.b64decode(face_rectangle))
            if "{}/{}".format(m_id, image_search_rank) in s:
                data = np.frombuffer(base64.b64decode(face_data), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                if h > 128 and w > 128:
                    try:
                        # run detector
                        results = detector.detect_face(img)

                        if results is not None:
                            total_boxes = results[0]
                            points = results[1]

                            bigbox_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in total_boxes])

                            # extract aligned face chips
                            chips = detector.extract_image_chips(img, points[bigbox_idx:bigbox_idx + 1], args.size,
                                                                 args.padding)
                            for i, chip in enumerate(chips):

                                if last_m_id != m_id:
                                    ab = "train"
                                    # let validation set has same class label as training set
                                    # see source code of pytorch's DatasetFolder
                                else:
                                    ab = "val" if random.random() > split_ratio else "train"
                                dd = os.path.join(dst_dir, ab, m_id)
                                os.makedirs(dd, exist_ok=True)
                                cv2.imwrite(os.path.join(dd, "{}.png".format(image_search_rank)),
                                            chip)
                                last_m_id = m_id

                    except Exception as e:
                        print(m_id, image_search_rank, e)

                    file_count = file_count + 1
                    if file_count % 1000 == 0:
                        print(file_count)


def read_clean_list(file_path):
    names = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            names.add(line[2:-5])
    return names


if __name__ == '__main__':
    main()
