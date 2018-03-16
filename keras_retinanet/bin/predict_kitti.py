import argparse
import os
import sys

import numpy as np
import keras
import tensorflow as tf

import cv2
from tqdm import tqdm

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.kitti_car import KITTICarGenerator
from ..utils.keras_version import check_keras_version
from ..utils.eval import evaluate, _get_detections
from ..utils.image import read_image_bgr, resize_image, preprocess_image
from ..utils.visualization import draw_detections
from ..models.resnet import custom_objects


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Predict script for a RetinaNet network.')

    parser.add_argument('--model',           help='Path to RetinaNet model.')
    parser.add_argument('--data-path',       help="path of kitti images", dest="data_path")
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',       help='Path for saving images with detections.', dest="save_path")
    parser.add_argument('--dets-path',       help='Path for saving dets', dest="dets_path")

    return parser.parse_args(args)


label_to_name = {
    0: "Car"
}

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.dets_path is not None and not os.path.exists(args.dets_path):
        os.makedirs(args.dets_path)

    # load the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    # print model summary
    print(model.summary())

    if args.data_path is None or not os.path.exists(args.data_path):
        print("can not find data_path")
        return

    paths = os.listdir(args.data_path)
    paths = [os.path.join(args.data_path, f) for f in paths]
    num_files = len(paths)

    for path in tqdm(paths):
        # bgr channel order
        raw_image = read_image_bgr(path)
        # preprocess
        image = preprocess_image(raw_image.copy())
        # resize
        image, scale = resize_image(image, min_side=768, max_side=2560)
        # added batch dim
        image = np.expand_dims(image, axis=0)

        # predict
        _, _, detections = model.predict_on_batch(image)

        # clip to image shape
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])

        # correct boxes for image scale
        detections[0, :, :4] /= scale

        # select scores from detections
        scores = detections[0, :, 4:]

        # select indices which have a score above the threshold
        indices = np.where(detections[0, :, 4:] > args.score_threshold)

        # select those scores
        scores = scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:args.max_detections]

        # select detections
        image_boxes      = detections[0, indices[0][scores_sort], :4]
        image_scores     = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
        image_detections = np.append(image_boxes, image_scores, axis=1)
        image_predicted_labels = indices[1][scores_sort]

        if args.save_path is not None:
            draw_detections(raw_image, detections[0, indices[0][scores_sort], :])
            cv2.imwrite(os.path.join(args.save_path, '{}.png'.format(os.path.basename(path))), raw_image)

        res_file = os.path.join(args.dets_path, "{}.txt".format(os.path.basename(path).split('.')[0]))
        with open(res_file, "w+") as f:
            for idx in range(image_detections.shape[0]):
                label = label_to_name[image_predicted_labels[idx]]
                xmin, ymin, xmax, ymax, score = image_detections[idx, :]
                f.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n" %
                    (label, xmin, ymin, xmax, ymax, score))

if __name__ == '__main__':
    main()
