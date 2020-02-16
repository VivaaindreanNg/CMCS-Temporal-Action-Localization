import os
import io
import torch
import zipfile
import argparse
import numpy as np
import subprocess
from PIL import Image
from scipy.signal import medfilt


def obtain_training_path(in_dir):
    '''
    Parameters
    ----------
    in_dir : 
        Path to directory which contains lists of directories for RGB & flow 
        of every videos.

    Returns
    -------
    train_vid_pth_list : 
        List containing paths to only training video's directories.
    '''
    
    train_vid_pth_list = []
    #Txt file containing anomalies videos (only training)
    anomaly_train_txt_pth = 'Anomaly_Train.txt'
    
    curr_dir = os.getcwd()
    curr_dir = os.path.join(os.getcwd(), anomaly_train_txt_pth)
    curr_dir = curr_dir.replace('\\', '/')
    
    
    f = open(anomaly_train_txt_pth, 'r')
    anomaly_train_vid_list = f.readlines()  
    for anomaly_train_vid in anomaly_train_vid_list:
        anomaly_train_vid = os.path.join(in_dir, anomaly_train_vid[:-5]).replace('\\', '/')
        train_vid_pth_list.append(anomaly_train_vid)
        
    path_to_train_normal_dir = os.path.join(in_dir, 'Training_Normal_Videos_Anomaly')
    train_normal_list = os.listdir(path_to_train_normal_dir)
    for normal_train_vid in train_normal_list:
        normal_train_vid = os.path.join(path_to_train_normal_dir, normal_train_vid).replace('\\', '/')
        train_vid_pth_list.append(normal_train_vid)
        
    train_vid_pth_list.sort()
        
    return train_vid_pth_list


def remove_short_clips(xx, min_length):

    x = np.array(xx)

    start = -1
    end = -1
    flag = False

    for i in range(len(x)):

        if not flag and x[i] == 1:
            start = i
            flag = True
        
        if flag and x[i] == 0:
            end = i
            flag = False

            if end - start < min_length:
                x[start:end] = 0

            start = -1
            end = -1

    if flag:
        end = i
        if end - start < min_length:
            x[start:end] = 0

    return x



def hard_negative_mining(train_vid_pth_list, out_dir, percentile_thrh):
    '''
    Parameters
    ----------
    train_vid_pth_list : 
        List containing paths to every training video's directories.
        
    out_dir: 
        Path to save static clips.
        
    percentile_thrh:
        Selection ratio for hard negative mining

    Returns
    -------
    None.

    '''
    fps = 30
    for video_dir in train_vid_pth_list:
        video_name = video_dir.split('/')[-1]
        if video_name.endswith('_x264'):
            print(video_name)
            save_file = os.path.join(out_dir, video_name + '.npz').replace('\\', '/')

            if save_file in os.listdir(out_dir):
                continue
            
            
            flowx_dir = os.path.join(video_dir, 'flow_x').replace('\\', '/')
            flow_x_files = [i for i in os.listdir(flowx_dir) if i.startswith('flow_x')]
            flowy_dir = os.path.join(video_dir, 'flow_y').replace('\\', '/')
            flow_y_files = [i for i in os.listdir(flowy_dir) if i.startswith('flow_y')]
    
            flow_x_files.sort()
            flow_y_files.sort()
    
            assert(len(flow_x_files) == len(flow_y_files))
    
            intensity_xy_mean = np.zeros((len(flow_x_files),))
    
            for frame_id in range(len(flow_x_files)):
                
                flow_x = Image.open(os.path.join(flowx_dir, flow_x_files[frame_id]))
                flow_y = Image.open(os.path.join(flowy_dir, flow_y_files[frame_id]))
    
                flow_x = np.array(flow_x).astype(float)
                flow_y = np.array(flow_y).astype(float)
    
                flow_x = (flow_x * 2 / 255) - 1  # -1, 1
                flow_y = (flow_y * 2 / 255) - 1
    
                flow_xy = np.sqrt(flow_x * flow_x + flow_y * flow_y)
    
                intensity_xy_mean[frame_id] = flow_xy.mean()
    
            intensity_xy_mean = np.log(intensity_xy_mean+0.0000001)
    
            threshold = np.percentile(intensity_xy_mean, percentile_thrh)
            thresholded = intensity_xy_mean < threshold
    
            filtered = medfilt(thresholded, kernel_size=(2*int(fps)-1))  # maybe shorter
    
            removed = remove_short_clips(filtered, 2*int(fps))  # maybe shorter
            
            #Remove static clip if it's too long or short
            bg_ratio = removed.sum() / removed.shape[0]
            if bg_ratio < 0.05 or bg_ratio > 0.30:
                print('Bad background: Ratio {}, {}'.format(bg_ratio, video_name))
                continue
            else:
                np.savez(save_file,
                    intensity=intensity_xy_mean,
                    mask=removed)
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_dir', type=str, 
                        help='Path to every video\'s rgb & flow directories')
    parser.add_argument('--out_dir', type=str,
                        help='Output directory for static background clips')
    parser.add_argument('--percentile_thrh', type=int, default=30,
                        help='Selection ratio for Hard Negative mining')  # Selection ratio for HN mining: 30% 
    args = parser.parse_args()
    
    
    print('Mining Hard Negatives from training videos')
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    train_vid_pth_list = obtain_training_path(args.in_dir)
    print('Total number of training videos: {}'.format(len(train_vid_pth_list)))
    hard_negative_mining(train_vid_pth_list, args.out_dir, args.percentile_thrh)
    
    
    
        
        