import os
import shutil
import argparse

def get_feature_path(feature_pth, dest_pth, sample_mode):
    
    #Txt file containing anomalies videos (only training)
    anomaly_train_txt_pth = 'Anomaly_Train.txt'
    test_txt_pth = 'Anomaly_Test.txt'
    train_rgb_dest = os.path.join(dest_pth, 'val-rgb-{}'.format(sample_mode))
    train_flow_dest = os.path.join(dest_pth, 'val-flow-{}'.format(sample_mode))
    test_rgb_dest = os.path.join(dest_pth, 'test-rgb-{}'.format(sample_mode))
    test_flow_dest = os.path.join(dest_pth, 'test-flow-{}'.format(sample_mode))
    train_rgb_list, train_flow_list = [], []
    test_rgb_list, test_flow_list = [], []
      
    if not os.path.exists(dest_pth):
        os.makedirs(dest_pth)
        
    if not os.path.exists(train_rgb_dest):
        os.makedirs(train_rgb_dest)
    
    if not os.path.exists(train_flow_dest):
        os.makedirs(train_flow_dest)
    
    if not os.path.exists(test_rgb_dest):
        os.makedirs(test_rgb_dest)
        
    if not os.path.exists(test_flow_dest):
        os.makedirs(test_flow_dest)
        
    curr_dir = os.path.join(os.getcwd(), anomaly_train_txt_pth)
    
    ##### Segregate input features (training) #####
    f = open(anomaly_train_txt_pth, 'r')
    anomaly_train_list = f.readlines()  
    for anomaly_train_vid in anomaly_train_list:
        video_name = anomaly_train_vid[:-5].split('/')[1]
        anomaly_rgb = '{}/{}-rgb.npz'.format(anomaly_train_vid[:-5], video_name)
        anomaly_flow = '{}/{}-flow.npz'.format(anomaly_train_vid[:-5], video_name)
        
        anomaly_rgb = os.path.join(feature_pth, anomaly_rgb)
        anomaly_flow = os.path.join(feature_pth, anomaly_flow)
        train_rgb_list.append(anomaly_rgb)
        train_flow_list.append(anomaly_flow)
        
    path_to_train_normal_dir = os.path.join(feature_pth, 'Training_Normal_Videos_Anomaly')
    train_normal_list = os.listdir(path_to_train_normal_dir)
    for normal_train_vid in train_normal_list:
        normal_train_rgb = '{}/{}-rgb.npz'.format(normal_train_vid, normal_train_vid)
        normal_train_flow = '{}/{}-flow.npz'.format(normal_train_vid, normal_train_vid)
        
        normal_train_rgb = os.path.join(path_to_train_normal_dir, normal_train_rgb)
        normal_train_flow = os.path.join(path_to_train_normal_dir, normal_train_flow)
        train_rgb_list.append(normal_train_rgb)
        train_flow_list.append(normal_train_flow)

        
    ##### Segregate input features (testing) #####
    f = open(test_txt_pth, 'r')
    test_list = f.readlines()
    for test_vid in test_list:
        video_name = test_vid[:-5].split('/')[1]
        test_rgb = '{}/{}-rgb.npz'.format(test_vid[:-5], video_name)
        test_flow = '{}/{}-flow.npz'.format(test_vid[:-5], video_name)
    
        test_rgb = os.path.join(feature_pth, test_rgb)
        test_flow = os.path.join(feature_pth, test_flow)
        test_rgb_list.append(test_rgb)
        test_flow_list.append(test_flow)
        
    train_rgb_list.sort()
    train_flow_list.sort()
    test_rgb_list.sort()
    test_flow_list.sort()
    
    return train_rgb_list, train_flow_list, test_rgb_list, test_flow_list, \
        train_rgb_dest, train_flow_dest, test_rgb_dest, test_flow_dest
        


def transfer(input_path_list, dst):
    for src in input_path_list:
        shutil.move(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_pth', type=str, 
                        help='Path to directory of i3d features')
    parser.add_argument('--dest_pth', type=str, 
                        help='Path to store i3d features for train & test (similar with the ones in config file)')
    parser.add_argument('--sample_mode', type=str,
                        help='Sample mode based on extracted i3d features (oversample, resize)')
    
    args = parser.parse_args()
    
    
    train_rgb_list, train_flow_list, test_rgb_list, test_flow_list, \
        train_rgb_dest, train_flow_dest, test_rgb_dest, \
            test_flow_dest = get_feature_path(
                feature_pth=args.feature_pth, 
                dest_pth=args.dest_pth, 
                sample_mode=args.sample_mode)

    transfer(train_rgb_list, train_rgb_dest)
    transfer(train_flow_list, train_flow_dest)
    transfer(test_rgb_list, test_rgb_dest)
    transfer(test_flow_list, test_flow_dest)
    
    
    
    
    
    