# coding: utf-8
import argparse
import os
import random
import shutil

import cv2
import mxnet as mx
import numpy as np

from mtcnn_detector import MtcnnDetector


def get_args():
    parser = argparse.ArgumentParser(description='align faces')
    parser.add_argument('--src', help='source directory')
    parser.add_argument('--dst', help='destination directory')
    parser.add_argument('--err', help='err directory')
    parser.add_argument('--workers', default=1, type=int, help='workers')
    parser.add_argument('--ngpus', default=-1, type=int, help='how many gpus to use')
    parser.add_argument('--size', default=128, type=int, help='output image is (size,size)')
    parser.add_argument('--padding', default=0.37, type=float, help='padding')
    parser.add_argument('--split', default=0, type=float, help='split ratio')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    src_dir = args.src
    if not os.path.exists(src_dir):
        raise ValueError("src dir not exist {}".format(src_dir))

    split_ratio = args.split

    dst_dir = os.path.abspath(args.dst)
    err_dir = os.path.abspath(args.err)

    num_gpus = args.ngpus
    if num_gpus == -1:
        num_gpus = len(mx.test_utils.list_gpus())
    if num_gpus == 0:
        ctx = mx.cpu(0)
    else:
        ctx = [mx.gpu(i) for i in range(num_gpus)]

    print("src dir={} dst dir={} err_dir={}".format(src_dir, dst_dir, err_dir))
    detector = MtcnnDetector(model_folder='model', ctx=ctx, num_worker=args.workers, accurate_landmark=False)

    for root, dirs, files in os.walk(src_dir):
        relpath = os.path.relpath(root, src_dir)
        # dd = os.path.join(dst_dir, relpath)
        ed = os.path.join(err_dir, relpath)
        class_data_written = False  # training
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.gif', '.png')):
                absfile = os.path.join(root, filename)
                success = False
                try:
                    # warning cv2.imread does not handle file names with unicode characters.
                    img = cv2.imread(absfile)

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
                            if split_ratio > 0:
                                if not class_data_written:
                                    ab = "train"
                                    class_data_written = True
                                    # let validation set has same class label as training set
                                    # see source code of pytorch's DatasetFolder
                                    os.makedirs(os.path.join(dst_dir, "val", relpath), exist_ok=True)
                                else:
                                    ab = "val" if random.random() > split_ratio else "train"
                                dd = os.path.join(dst_dir, ab, relpath)
                                os.makedirs(dd, exist_ok=True)
                                cv2.imwrite(os.path.join(dd, os.path.splitext(filename)[0] + ".png"),
                                            chip)
                                class_data_written = True
                            else:
                                dd = os.path.join(dst_dir, relpath)
                                os.makedirs(dd, exist_ok=True)
                                cv2.imwrite(os.path.join(dd, os.path.splitext(filename)[0] + ".png"),
                                            chip)
                            success = True

                except Exception as e:
                    print(relpath, filename, e)
                    pass

                if not success:
                    os.makedirs(ed, exist_ok=True)
                    shutil.copyfile(absfile, os.path.join(ed, filename))


if __name__ == '__main__':
    main()
