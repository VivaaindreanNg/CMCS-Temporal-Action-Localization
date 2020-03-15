import numpy as np 
import pandas as pd 
import pdb
import os
import argparse

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from demo import metric_scores
from utils import load_config_file, ucf_crime_old_cls_names


def prepare_gt(gtpth):
    gt_list = []
    fps = 30
    f = open(gtpth, 'r')
    for line in f.readlines():
        line2 = []
        line = line.replace('.mp4', '')
        line = line.split('  ')
        # Skip Normal videos
        if line[0].startswith('Normal'):
            continue
        gt_list.append(line)
    
    df = pd.DataFrame(gt_list)
    df.columns = ['videoname', 'cls', 'start1', 'end1', 'start2', 'end2', '_']
    df = df.drop(columns=['_'])

    f.close()
    return df

def prepare_detections(detlist):
    df = pd.DataFrame(detlist)
    df.columns = ['videoname', 'start', 'end', 'cls', 'conf'] 
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--config-file', type=str,
        help='Path to config files'
    )
    parser.add_argument(
        '--to-rgb', type=str, 
        help='Path to dir containing rgb directories of every video'
    )
    parser.add_argument('--test-subset-name', type=str)
    args = parser.parse_args()

    print(args.config_file)
    print(args.test_subset_name)
    print(args.to_rgb)

    all_params = load_config_file(args.config_file)
    locals().update(all_params)

    for run_idx in range(all_params['train_run_num']):

        for cp_idx, check_point in enumerate(all_params['check_points']):

            for mod_idx, modality in enumerate([
                'both', 'rgb', 'flow', 'late-fusion']):
                cas_dir = os.path.join(
                    'cas-features',
                    '{}-run-{}-{}-{}'.format(
                        all_params['experiment_naming'],
                        run_idx,
                        check_point,
                        modality
                    ))

    
               
                list_video_names = os.listdir(cas_dir)
                vid_name = [v.split('/')[-1][:-4] for v in list_video_names] # Obtain video's name

                

                #To contain path of every single video's rgb dir (To obtain frame count per video)
                rgb_pth_list = []
                for clss in range(1, all_params['action_class_num']+1):
                    for video_test_set in vid_name:
                        if video_test_set.startswith(ucf_crime_old_cls_names[clss]):
                            video_test_set = os.path.join(video_test_set, 'rgb')
                            to_rgb = os.path.join(args.to_rgb, ucf_crime_old_cls_names[clss])
                            to_rgb = os.path.join(to_rgb, video_test_set)
                            rgb_pth_list.append(to_rgb)
                            to_rgb = 'D:/Input_Frame_dirs'


                #### Obtain ground truth  ####
                gt_file_pth = all_params['file_paths'][args.test_subset_name]['anno_dir']
                ground_truth = os.listdir(gt_file_pth)[0]
                ground_truth = os.path.join(gt_file_pth, ground_truth)
                gtdf = prepare_gt(ground_truth)
                

                #### Obtain predicted (average of all branches) scores ####

                All_det, All_GT = [], []

                for vid, j in zip(list_video_names, rgb_pth_list):
                    scores, frame_cnt, _, detected_list = metric_scores(
                        os.path.join(cas_dir, vid), 
                        j, 
                        **all_params
                    )
                    

                    ############## Get frame-level scores for average of all branches ##############
                    if scores is not None:
                        frame_lvl_score = scores.flatten().tolist()
                    
                    assert(len(frame_lvl_score) == frame_cnt)
                    All_det += frame_lvl_score
                    
                    ################ Get frame-level ground truth ################
                    gt_frame_label = np.zeros((1, frame_cnt))
                    gt_entries = gtdf[gtdf['videoname'] == vid[:-4]]
                    
                    if int(gt_entries['start1']) != -1:
                        start1 = int(gt_entries['start1']) 
                        end1 = int(gt_entries['end1']) 
                        gt_frame_label[:, start1:end1] = 1

                    if int(gt_entries['start2']) != -1:
                        start2 = int(gt_entries['start2'])
                        end2 = int(gt_entries['end2'])
                        gt_frame_label[:, start2:end2] = 1

                    gt_frame_label = gt_frame_label.flatten().tolist()
                    assert(len(gt_frame_label) == frame_cnt)
                    All_GT += gt_frame_label
                
                assert(len(All_GT) == len(All_det))
                #################### Calculate frame-level ROC ####################   
                #Calculate AUC for frame-level
                auc = roc_auc_score(All_GT, All_det)
                print('({}) AUC for ROC: {:.4f}'.format(
                    modality,
                    auc
                ))
    
    

        

    