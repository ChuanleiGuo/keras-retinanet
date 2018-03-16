import argparse
import os
import sys

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.kitti_car import KITTICarGenerator
from ..utils.keras_version import check_keras_version
from ..utils.eval import evaluate, _get_detections
from ..models.resnet import custom_objects


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    if args.dataset_type == "kitti_car":
        generator = KITTICarGenerator(
            args.kitti_path,
            "training",
            shuffle_groups=False
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    kitti_parser = subparsers.add_parser("kitti_car")
    kitti_parser.add_argument("kitti_path", help="Path to dataset directory (ie. /tmp/KITTI).")

    parser.add_argument('model',             help='Path to RetinaNet model.')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',       help='Path for saving images with detections.')
    parser.add_argument('--dets-path',       help='Path for saving dets', required=True)

    return parser.parse_args(args)


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

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    # print model summary
    print(model.summary())

    # start predict
    detections = _get_detections(
        generator,
        model,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path
    )

    import pdb; pdb.set_trace()
    for idx in range(detections.shape[0]):
        for cls_idx in range(detections[idx].shape[0]):
            dets = detections[idx, cls_idx, :, :]
            for box_id in range(dets.shape[0]):
                xmin, ymin, xmax, ymax, score = dets[box_id]
                with open(os.path.join(args.dets_path, "%06d.txt" % (idx)), "a+") as f:
                    f.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n" % \
                        (generator.label_to_name(cls_idx),xmin,ymin,xmax,ymax,score))

    print(detections)


if __name__ == '__main__':
    main()
