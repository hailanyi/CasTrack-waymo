from tracker.tracker import Tracker3D
from tracker.config import creat_config
import numpy as np
import argparse
from tqdm import trange
import os
import pickle
import copy
from tracker.box_op import register_bbs
from dataset.waymo_eval_track import OpenPCDetWaymoDetectionMetricsEstimator

def tracking_one_cls(src_detections, config, this_cls='vehicle', infos=None, seed=0):

    assert len(src_detections) == len(infos)
    print('tracking cls: ', this_cls)
    track_config = config[this_cls]
    detections = copy.deepcopy(src_detections)
    pre_seq_id = detections[0]['seq_id']

    tracker = Tracker3D(box_type="OpenPCDet", tracking_features=False, config = track_config)
    tracker.label_seed = seed+1
    for i in trange(len(detections)):
        det = detections[i]
        info = infos[i]
        pose = info['pose']
        cur_seq_id = det['seq_id']

        if pre_seq_id!=cur_seq_id:
            tracker.dead_trajectories.clear()
            tracker.active_trajectories.clear()
            pre_seq_id = cur_seq_id

        cur_score = det['score']
        cur_box = det['boxes_lidar']
        cur_name = det['name']
        input_score_mask = cur_score>=track_config.input_score
        cur_score = cur_score[input_score_mask]
        cur_box = cur_box[input_score_mask]
        cur_name = cur_name[input_score_mask]
        name_mask = cur_name==this_cls

        cur_box = cur_box[name_mask]
        cur_score = cur_score[name_mask]

        cur_box, ids, cur_score = tracker.tracking(bbs_3D=cur_box,
                                                   scores=cur_score,
                                                   pose=pose,
                                                   timestamp=i
                                                   )
        new_pose = np.mat(pose).I
        cur_box = register_bbs(cur_box,new_pose)

        out_score_mask = cur_score>track_config.post_score

        cur_score=cur_score[out_score_mask]
        ids=ids[out_score_mask]
        cur_box=cur_box[out_score_mask]

        det['score'] = cur_score
        det['boxes_lidar'] = cur_box
        det['name'] = np.array([this_cls]*len(cur_box))
        det['obj_ids'] = ids

    return detections, tracker.label_seed

def merge_eval_save(all_tracked_results, config, logger, infos):
    all_results = copy.deepcopy(all_tracked_results[config.tracking_class[0]])

    for i in range(len(all_results)):
        names_list = []
        box_list = []
        score_list = []
        ids_list = []
        for cls in config.tracking_class:
            frame = all_tracked_results[cls][i]
            if len(frame['name'])>0:
                names_list.append(frame['name'])
                box_list.append(frame['boxes_lidar'])
                score_list.append(frame['score'])
                ids_list.append(frame['obj_ids'])
        if len(names_list)!=0:
            all_results[i]['name'] = np.concatenate(names_list)
            all_results[i]['boxes_lidar'] = np.concatenate(box_list)
            all_results[i]['score'] = np.concatenate(score_list)
            all_results[i]['obj_ids'] = np.concatenate(ids_list)
        else:
            all_results[i]['name'] = np.zeros(shape=(0,))
            all_results[i]['boxes_lidar'] = np.zeros(shape=(0,7))
            all_results[i]['score'] = np.zeros(shape=(0,))
            all_results[i]['obj_ids'] = np.zeros(shape=(0,))
    with open(os.path.join(config.save_path,'result.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    if config.split!='test':
        eval = OpenPCDetWaymoDetectionMetricsEstimator()

        gt_infos_dst = []
        for idx in range(0, len(infos)):
            cur_info = infos[idx]['annos']
            sample_idx = infos[idx]['point_cloud']['sample_idx']
            seq_idx = infos[idx]['point_cloud']['lidar_sequence']
            cur_info['frame_id'] = sample_idx
            cur_info['seq_id'] = seq_idx
            gt_infos_dst.append(cur_info)

        ap_dict = eval.waymo_evaluation(
            all_results, gt_infos_dst, class_name=config.tracking_class, distance_thresh=1000, fake_gt_infos=False
        )

        ap_result_str = '\n'
        for key in ap_dict:
            ap_dict[key] = ap_dict[key][0]
            ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

        logger.info(ap_result_str)


def tracking_all(config, logger):

    detections_path = os.path.join(config.detections_path, config.split, 'result.pkl')
    info_path = os.path.join(config.info_path, 'waymo_infos_'+config.split+'.pkl')
    detections = pickle.load(open(detections_path, 'rb'))
    infos = pickle.load(open(info_path,'rb'))
    logger.info('all tracking frames: %d'%len(detections))

    all_tracked_results = {}
    seed = 0
    for cls in config.tracking_class:

        cur_results, seed = tracking_one_cls(detections, config, cls, infos, seed)
        all_tracked_results[cls] = cur_results

    merge_eval_save(all_tracked_results, config, logger, infos)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="config/config.yaml",
                        help='specify the config for tracking')
    args = parser.parse_args()
    yaml_file = args.cfg_file

    config, logger = creat_config(yaml_file)
    tracking_all(config, logger)








