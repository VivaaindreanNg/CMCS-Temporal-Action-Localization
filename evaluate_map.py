import pandas as pd
import os
import torch
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import smooth
from utils import detect_with_thresholding
from utils import get_dataset, normalize, interpolate
from utils import mask_to_detections, load_config_file
from utils import output_detections_ucf_crime
from utils import ucf_crime_old_cls_names, ucf_crime_old_cls_indices
from utils import prepare_gt, segment_iou, load_config_file
from collections import OrderedDict

import pdb


def softmax(x, dim):
    x = F.softmax(torch.from_numpy(x), dim=dim)
    return x.numpy()


def prepare_detections(detpth):
    f = open(detpth)
    det_list = []
    for i in f.readlines():
        i = i.replace('\n', '')
        i = i.split(' ')
        det_list.append(i)
    
    df = pd.DataFrame(det_list)
    df.columns = ['videoname', 'start', 'end', 'cls', 'conf']
    f.close()
    return df

def interpolated_prec_rec(prec, rec):
    """
    Interpolated AP - VOCdevkit from VOC 2011. 
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds):
    """
    Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest IoU score is matched as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : 
        Data frame containing the ground truth instances.
        Required fields: ['videoname', 'start', 'end']
    prediction : 
        Data frame containing the prediction instances.
        Required fields: ['videoname', 'start', 'end', 'conf']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """

    prediction = prediction.reset_index(drop=True)
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1 
    # Sort predictions by decreasing (confidence) score order.
    sort_idx = prediction['conf'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('videoname')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated with predicted video.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['videoname'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['start', 'end']].values, this_gt[['start', 'end']].values)

        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1] 
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    ap = np.zeros(len(tiou_thresholds))
    rec, prec = None, None
    for tidx in range(len(tiou_thresholds)):
        # Computing prec-rec (per-class basis at every individual tiou_thresholds)
        this_tp = np.cumsum(tp[tidx,:]).astype(np.float)
        this_fp = np.cumsum(fp[tidx,:]).astype(np.float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)

    return ap


def eval_ap(iou, clss, gt, prediction):
    ap = compute_average_precision_detection(gt, prediction, iou)
    return ap[0]


def detect(
        dataset_dicts,
        cas_dir,
        subset,
        out_file_name,
        global_score_thrh,
        metric_type,
        thrh_type,
        thrh_value,
        interpolate_type,
        proc_type,
        proc_value,
        sample_offset,
        weight_inner,
        weight_outter,
        weight_global,
        att_filtering_value=None,
    ):

    assert (metric_type in ['score', 'multiply', 'att-filtering'])
    assert (thrh_type in ['mean', 'max'])
    assert (interpolate_type in ['quadratic', 'linear', 'nearest'])
    assert (proc_type in ['dilation', 'median'])

    out_detections = []

    dataset_dict = dataset_dicts[subset]

    for video_name in dataset_dict.keys():

        cas_file = video_name + '.npz'
        cas_data = np.load(os.path.join(cas_dir, cas_file))

        avg_score = cas_data['avg_score']
        att_weight = cas_data['weight']
        branch_scores = cas_data['branch_scores']
        global_score = cas_data['global_score']

        fps = dataset_dict[video_name]['frame_rate']
        frame_cnt = dataset_dict[video_name]['frame_cnt']
        duration = frame_cnt/fps

        global_score = softmax(global_score, dim=0)

        ################ Thresholding ################
        for class_id in range(all_params['action_class_num']):

            if global_score[class_id] <= global_score_thrh:
                continue

            if metric_type == 'score':

                metric = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
                #metric = smooth(metric)
                metric = normalize(metric)

            elif metric_type == 'multiply':

                _score = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
                metric = att_weight * _score
                #metric = smooth(metric)
                metric = normalize(metric)

            elif metric_type == 'att-filtering':
                assert (att_filtering_value is not None)

                metric = softmax(avg_score, dim=1)[:, class_id:class_id + 1]
                #metric = smooth(metric)
                metric = normalize(metric)
                metric[att_weight < att_filtering_value] = 0
                metric = normalize(metric)

            #########################################

            metric = interpolate(metric[:, 0],
                                    all_params['feature_type'],
                                    frame_cnt,
                                    all_params['base_sample_rate']*all_params['sample_rate'],
                                    snippet_size=all_params['base_snippet_size'],
                                    kind=interpolate_type)

            metric = np.expand_dims(metric, axis=1)

            mask = detect_with_thresholding(metric, thrh_type, thrh_value,
                                            proc_type, proc_value)

            temp_out = mask_to_detections(mask, metric, weight_inner,
                                            weight_outter)

            #########################################

            for entry in temp_out:

                entry[2] = class_id
                entry[3] += global_score[class_id] * weight_global

                entry[0] = (entry[0] + sample_offset) / fps
                entry[1] = (entry[1] + sample_offset) / fps

                entry[0] = max(0, entry[0])
                entry[1] = max(0, entry[1])
                entry[0] = min(duration, entry[0])
                entry[1] = min(duration, entry[1])

            #########################################

            for entry_id in range(len(temp_out)):
                temp_out[entry_id] = [video_name] + temp_out[entry_id]

            out_detections += temp_out

    output_detections_ucf_crime(out_detections, out_file_name)


    return out_detections




if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config-file', type=str)
    parser.add_argument('--test-subset-name', type=str)
    args = parser.parse_args()

    print(args.config_file)
    print(args.test_subset_name)

    all_params = load_config_file(args.config_file)
    locals().update(all_params)
    
    odict = OrderedDict()

    test_dataset_dict = get_dataset(
        dataset_name=all_params['dataset_name'],
        subset=args.test_subset_name,
        file_paths=all_params['file_paths'],
        sample_rate=all_params['sample_rate'],
        base_sample_rate=all_params['base_sample_rate'],
        action_class_num=all_params['action_class_num'],
        modality='both',
        feature_type=all_params['feature_type'],
        feature_oversample=False,
        temporal_aug=False,
    )

    dataset_dicts = {'test': test_dataset_dict}

    # Obtain ground truth
    gt_file_pth = all_params['file_paths'][args.test_subset_name]['anno_dir']
    ground_truth = os.listdir(gt_file_pth)[0]
    gtpth = os.path.join(gt_file_pth, ground_truth)
    gtdf = prepare_gt(gtpth)


    summary_file = './outputs/summary-{}.json'.format(all_params['experiment_naming'])

    for run_idx in range(all_params['train_run_num']):

        for cp_idx, check_point in enumerate(all_params['check_points']):

            for mod_idx, modality in enumerate(
                ['both', 'rgb', 'flow', 'late-fusion']):

                cas_dir = os.path.join(
                    'cas-features',
                    '{}-run-{}-{}-{}'.format(all_params['experiment_naming'], run_idx,
                                             check_point, modality))

                pred_dir = os.path.join('outputs', 'predictions')

                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)


                test_pred_file = os.path.join(
                    pred_dir,
                    '{}-run-{}-{}-{}-test'.format(all_params['experiment_naming'], 
                                                  run_idx,
                                                  check_point, 
                                                  modality))
                
                # test_outs = [vid_name, start, end, predict_class, confidence_score]
                test_outs = detect(dataset_dicts, cas_dir, 'test', test_pred_file,
                                   **all_params['detect_params'])

                iou_range = [.1, .2, .3, .4, .5]
                odict[modality] = {}
                for IoU_idx, IoU in enumerate(iou_range):

                    if len(test_outs) != 0:
                        # Obtain detection predictions
                        detdf = prepare_detections(test_pred_file)

                        # Rearranging the ground-truth and predictions based on class
                        gt_by_cls, det_by_cls = [], []
                        for clss in range(1, all_params['action_class_num']+1):
                            gt_by_cls.append(gtdf[gtdf['cls'] == clss].reset_index(drop=True).drop('cls', 1))
                            det_by_cls.append(detdf[detdf['videoname'].str.contains(ucf_crime_old_cls_names[clss])].drop(columns=['cls']))
                        
                        ap_values = []
                        mAP = 0
                        for clss in range(all_params['action_class_num']):
                            #ap per-class basis
                            ap = eval_ap([IoU], clss, gt_by_cls[clss], det_by_cls[clss])
                            ap_values.append(ap)
                        
                        # calculate mean AP across all classes (for every IoU value)
                        for i in ap_values:
                            mAP += i
                        mAP = mAP/len(ap_values)

                    else:
                        print('Empty Detections')

                    odict[modality][IoU] = '{:.4f}'.format(mAP)

    for k, v in odict.items():
        print(k, v)                

    json = json.dumps(odict)
    f = open(summary_file, 'w')
    f.write(json)
    f.close()

