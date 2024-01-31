# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from mmdet.evaluation.functional import bbox_overlaps
import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
import json
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=int,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta
    
    #seat detect
    with open(f"{args.video}.json",'r') as f:
        bbox_seat = {a['label']:np.array(a['points']).flatten() for a in json.load(f)["shapes"]}
        seat_labels = sorted(bbox_seat)
        bbox_seat = np.stack([bbox_seat[l] for l in seat_labels],axis=0)
    visualizer.dataset_meta['classes'] = seat_labels


    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
    frame_nums = list(np.arange(0, len(video_reader),30))
    for i in track_iter_progress(frame_nums):
        frame = video_reader.get_frame(i)
        result = inference_detector(model, frame, test_pipeline=test_pipeline)
        labels = np.array(result.pred_instances.labels.squeeze().cpu())
        bbox_pred = np.array(result.pred_instances.bboxes.squeeze().cpu())
        overlaps = bbox_overlaps(bbox_pred,bbox_seat)
        keep = np.any(overlaps>0.3,axis=1) & (labels==0)
        seat_label = np.argmax(overlaps,axis=1)
        result.pred_instances = result.pred_instances[keep]
        result.pred_instances.labels = seat_label[keep]
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=args.score_thr)
        frame = visualizer.get_image()

        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
