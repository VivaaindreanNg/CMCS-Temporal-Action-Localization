import cv2
import os
import numpy as np
#from glob import glob
import shutil
from multiprocessing import Pool
import argparse



def gen_video(video_path, rgbflow_path):
    '''
    Parameters
    ----------
    video_path : The (root)path to input video directory
    rgbflow_path : The path to destination directory for saving RGB and optical flow.

    Returns
    -------
    list_of_videos : List of every single video names
    list_rgb_dirs : The path to every video's RGB dir
    list_flowx_dirs : The path to every video's flow_x dir
    list_flowy_dirs : The path to every video's flow_y dir
    list_video_paths: The path to every single input video
    '''
    list_video_paths = []
    list_of_videos = []
    list_rgb_dirs = []
    list_flowx_dirs = []
    list_flowy_dirs = []
    out_dir_path = []

    if not os.path.exists(rgbflow_path):
        os.makedirs(rgbflow_path)
        
    # For every video file, get its path
    for root, dirs, files in os.walk(video_path):
        for f in files:
            if f.endswith('.mp4'):
                filepath = os.path.join(root, f)
                filepath = os.path.abspath(filepath)
                filepath = filepath.replace('\\', '/') # Enable cv2 to properly capture video path
                list_video_paths.append(filepath)
                #list_of_videos.append(f)
    
    # For every video file, create corresponding (output) directories for RGB & flow
    out_dir_path = [os.path.relpath(v, video_path) for v in list_video_paths]
    out_dir_path = [os.path.join(rgbflow_path, o).replace('\\', '/').split('.')[0] for o in out_dir_path]
    for video_dir in out_dir_path:
        os.makedirs(os.path.join(video_dir, 'rgb'))
        os.makedirs(os.path.join(video_dir, 'flow_x'))
        os.makedirs(os.path.join(video_dir, 'flow_y'))     
        
        list_rgb_dirs.append(os.path.join(video_dir, 'rgb'))
        list_flowx_dirs.append(os.path.join(video_dir, 'flow_x'))
        list_flowy_dirs.append(os.path.join(video_dir, 'flow_y'))       
    #list_of_videos.sort()
    
    return list_rgb_dirs, list_flowx_dirs, list_flowy_dirs, list_video_paths


def extract_rgb(input_vid_path, dir_name):
    '''
    Parameters
    ----------
    input_vid_path : The (root)path to input video directory
    dir_name : The path to rgb directory that corresponds with every video.

    Returns
    -------
    None
    '''
    cap = cv2.VideoCapture(input_vid_path)
    #cap.set(cv2.CAP_PROP_FPS, 25)
    if not cap.isOpened():
        print('Could not capture video at {}'.format(input_vid_path))
    
    frame_count = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if ret == True:
            # Save frame
            save_rgb_path = os.path.join(dir_name, 'rgb{:07d}.jpg'.format(frame_count))
            cv2.imwrite(save_rgb_path, frame)
            frame_count+=1
            
        else:
            break
            
    
    print('Obtained RGB for {}'.format(input_vid_path))        
    cap.release()
    cv2.destroyAllWindows
        


def compute_TVL1flow(args):
    '''
    Parameters (args)
    ----------
    in_path : The path to every video's rgb directory
    flow_x_path : The path to flow_x directory that corresponds with every video.
    flow_y_path : The path to flow_x directory that corresponds with every video.

    Returns
    -------
    None
    '''
    bound = 20
    in_path, flow_x_path, flow_y_path = args
    rgb_frames = os.listdir(in_path)
    rgb_frames.sort()
    flow = []
    tv_l1 = cv2.createOptFlow_DualTVL1()

    def obtain_flows(in_path, rgb_frames):
        prev_frame = cv2.imread(os.path.join(in_path, rgb_frames[0]))
        prev_frame = cv2.UMat(prev_frame) # Convert to UMat to speed up computation by small factor
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_gray = cv2.UMat(prev_gray)

        for i, frames in enumerate(rgb_frames):
            curr_frame = cv2.imread(os.path.join(in_path, frames))
            curr_frame = cv2.UMat(curr_frame)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.UMat(curr_gray)

            tvl1_flow = tv_l1.calc(prev_gray, curr_gray, None)
            tvl1_flow = cv2.UMat.get(tvl1_flow)
            assert (tvl1_flow.dtype == np.float32)
            tvl1_flow = (tvl1_flow + bound) * (255.0 / (2*bound))
            tvl1_flow = np.round(tvl1_flow).astype(int)
            tvl1_flow[tvl1_flow >= 255] = 255
            tvl1_flow[tvl1_flow <= 0] = 0
            prev_gray = curr_gray
            flow.append(tvl1_flow)

        for i, flow_val in enumerate(flow):
            cv2.imwrite(os.path.join(flow_x_path.format('u'), "flow_x{:07d}.jpg".format(i)), flow_val[:, :, 0])
            cv2.imwrite(os.path.join(flow_y_path.format('v'), "flow_y{:07d}.jpg".format(i)), flow_val[:, :, 1])
            
    
    max_frame_per_dir = 15000 # Smaller groups
    # If total number of frames higher, incrementally calculate flows 
    if len(rgb_frames) > max_frame_per_dir:
        for groups in range(int(np.ceil(len(rgb_frames)/max_frame_per_dir))):
            rgb_frames_tmp = rgb_frames[groups*max_frame_per_dir:(groups+1)*max_frame_per_dir]
            obtain_flows(in_path, rgb_frames_tmp)
        print('Obtained flows for {}'.format(in_path)) 

    # Else, directly compute flows
    else:
        obtain_flows(in_path, rgb_frames)
        print('Obtained flows for {}'.format(in_path)) 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, 
                        help='Path to entire input video containing UCF-Crimes')
    parser.add_argument('--rgb_flow', type=str, 
                        help="Path to save every video's optical flow & RGB")
    parser.add_argument('--num_worker', type=int, default=2, 
                        help='For multiprocessing')

    args = parser.parse_args()
    video_path = args.video_dir
    rgb_flow = args.rgb_flow
    num_worker = args.num_worker
    
    list_rgb_dirs, list_flowx_dirs, list_flowy_dirs, list_video_paths = gen_video(video_path, rgb_flow)

    # Firstly, extract RGB frames for every video
    for i, j in zip(list_video_paths, list_rgb_dirs):
        extract_rgb(i, j)
    
    # Subsequently, extract vertical & horizontal flows 
    pool = Pool(num_worker)
    pool.map(compute_TVL1flow, zip(list_rgb_dirs, list_flowx_dirs, list_flowy_dirs))
    
    
    
    




