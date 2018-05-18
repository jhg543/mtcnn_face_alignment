# coding: utf-8
import argparse
import os
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
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    src_dir = args.src
    if not os.path.exists(src_dir):
        raise ValueError("src dir not exist {}".format(src_dir))

    dst_dir = os.path.abspath(args.dst)
    os.makedirs(dst_dir, exist_ok=True)
    err_dir = os.path.abspath(args.err)
    os.makedirs(err_dir, exist_ok=True)

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
        dd = os.path.join(dst_dir, relpath)
        ed = os.path.join(err_dir, relpath)
        dd_exist = os.path.exists(dd)
        ed_exist = os.path.exists(ed)
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.gif', '.png')):
                absfile = os.path.join(root, filename)
                success = False
                try:
                    img = cv2.imread(absfile)

                    # run detector
                    results = detector.detect_face(img)

                    if results is not None:

                        total_boxes = results[0]
                        points = results[1]

                        bigbox_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in total_boxes])

                        # extract aligned face chips
                        chips = detector.extract_image_chips(img, points[bigbox_idx:bigbox_idx + 1], args.size, args.padding)
                        for i, chip in enumerate(chips):
                            if not dd_exist:
                                os.makedirs(dd, exist_ok=True)
                                dd_exist = True
                            cv2.imwrite(os.path.join(dd, os.path.splitext(filename)[0] + ".png"), chip)
                            success = True

                except:
                    pass

                if not success:
                    if not ed_exist:
                        os.makedirs(ed, exist_ok=True)
                        ed_exist = True
                    shutil.copyfile(absfile, os.path.join(ed, filename))

                    pass


if __name__ == '__main__':
    main()
