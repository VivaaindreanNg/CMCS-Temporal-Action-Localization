import numpy as np 
import pandas as pd 
import pdb
import os
import shutil
import argparse
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from utils import metric_scores
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
        '--test-subset-name', type=str
    )

    parser.add_argument(
        '--save-auc', type=str,
        help='Path to store AUC (.mat files) for each modality including graph'
    )
    args = parser.parse_args()

    print(args.config_file)
    print(args.test_subset_name)
    print(args.save_auc)

    if os.path.exists(args.save_auc):
        shutil.rmtree(args.save_auc)
    os.makedirs(args.save_auc)

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


                #### Obtain ground truth  ####
                gt_file_pth = all_params['file_paths'][args.test_subset_name]['anno_dir']
                ground_truth = os.listdir(gt_file_pth)[0]
                ground_truth = os.path.join(gt_file_pth, ground_truth)
                gtdf = prepare_gt(ground_truth)
                

                #### Obtain predicted (average of all branches) scores ####

                All_det, All_GT = [], []
                auc_dict = {}

                for vid in list_video_names:
                    scores, frame_cnt, _, _ = metric_scores(
                        os.path.join(cas_dir, vid), 
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

                # Save info for plotting AUC graph
                fpr, tpr, _ = roc_curve(All_GT, All_det)
                
                auc_dict['modality'] = modality
                auc_dict['X'] = np.reshape(fpr, (fpr.size, 1))
                auc_dict['Y'] = np.reshape(tpr, (tpr.size, 1))
                auc_dict['AUC'] = auc
                savemat(
                    file_name=args.save_auc + '{}.mat'.format(modality), 
                    mdict=auc_dict)


    # Plot AUC graph
    color_list = ['blue', 'cyan', 'pink', 'green']
    auc_category = os.listdir(args.save_auc)

    for i in range(len(auc_category)):
        path_each_auc = os.path.join(args.save_auc, auc_category[i])
        get_auc = loadmat(path_each_auc)

        x_axis = get_auc['X']
        y_axis = get_auc['Y']
        auc = np.reshape(get_auc['AUC'], -1)
        auc *= 100
        auc = "%.2f" % auc
        plt.plot(x_axis, y_axis, color=color_list[i], 
             label='{} ({})'.format(auc_category[i][:-4], auc))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Comparison of AUC score (modalities)")
        plt.grid(True)
        plt.legend(loc="lower right")
    auc_plot_pth = args.save_auc + 'auc.png'
    plt.savefig(auc_plot_pth)
    


    
    

        

    