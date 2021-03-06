import matlab.engine  # Must import matlab.engine first
import os
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


from model import BackboneNet
from dataset import SingleVideoDataset
from utils import get_dataset, load_config_file

#import pdb

device = torch.device('cuda')


def get_diversity_loss(scores):

    assert (len(scores) > 1)

    softmax_scores = [F.softmax(i, dim=2) for i in scores]

    S1 = torch.stack(softmax_scores).permute(1, 3, 0, 2)
    S2 = torch.stack(softmax_scores).permute(1, 3, 2, 0)
    S1_norm = S1.norm(p=2, dim=3, keepdim=True)  # + 1e-6 careful here
    S2_norm = S2.norm(p=2, dim=2, keepdim=True)  #

    R = torch.matmul(S1, S2) / (torch.matmul(S1_norm, S2_norm) + 1e-6)

    I = torch.eye(len(scores)).to(device)
    I = I.repeat((R.shape[0], R.shape[1], 1, 1))

    pair_num = len(scores) * (len(scores) - 1)

    loss_div = F.relu(R - I).sum(-1).sum(-1) / pair_num
    loss_div = loss_div.mean()

    return loss_div


def get_norm_regularization(scores):

    video_len = scores[0].shape[1]

    assert (video_len > 0)

    S_raw = torch.stack(scores).permute(1, 3, 0, 2)
    S_raw_norm = S_raw.norm(p=1, dim=3) / video_len

    deviations = S_raw_norm - S_raw_norm.mean(dim=2, keepdim=True).repeat(
        1, 1, S_raw_norm.shape[2])

    loss_norm = torch.abs(deviations).mean()

    return loss_norm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--train-subset-name', type=str)
    parser.add_argument('--test-subset-name', type=str)

    parser.add_argument('--test-log', dest='test_log', action='store_true')
    parser.add_argument('--no-test-log', dest='test_log', action='store_false')
    parser.set_defaults(test_log=True)

    args = parser.parse_args()

    print(args.config_file)
    print(args.train_subset_name)
    print(args.test_subset_name)
    print(args.test_log)
    
    all_params = load_config_file(args.config_file)
    locals().update(all_params)
    

    def test(model, loader, modality):

        assert (modality in ['both', 'rgb', 'flow'])

        pred_score_dict = {}
        label_dict = {}

        correct = 0
        total_cnt = 0
        total_loss = {
            'cls': 0,
            'div': 0,
            'norm': 0,
            'sum': 0,
        }

        criterion = nn.CrossEntropyLoss(reduction='mean')

        with torch.no_grad():

            model.eval()

            for _, data in enumerate(loader):  # No shuffle

                video_name = data['video_name'][0]
                label = data['label'].to(device)
                weight = data['weight'].to(device).float()

                if label.item() == all_params['action_class_num']:
                    continue
                else:
                    total_cnt += 1

                if modality == 'both':
                    if data['rgb'].shape[1] != 0 and data['flow'].shape[1] != 0:
                        rgb = data['rgb'].to(device).squeeze(0)
                        flow = data['flow'].to(device).squeeze(0)
                        model_input = torch.cat([rgb, flow], dim=2)
                elif modality == 'rgb':
                    if data['rgb'].shape[1] != 0:
                        model_input = data['rgb'].to(device).squeeze(0)
                else:
                    if data['flow'].shape[1] != 0:
                        model_input = data['flow'].to(device).squeeze(0)

                if model_input.shape[-1] == all_params['feature_dim']:
                    model_input = model_input.transpose(2, 1)
                #Both
                if modality == 'both' and model_input.shape[-1] == all_params['feature_dim']*2:
                    model_input = model_input.transpose(2, 1)
                _, _, out, scores, _ = model(model_input)

                out = out.mean(0, keepdim=True)

                loss_cls = criterion(out, label) * weight
                total_loss['cls'] += loss_cls.item()

                if all_params['diversity_reg']:

                    loss_div = get_diversity_loss(scores) * weight
                    loss_div = loss_div * all_params['diversity_weight']

                    loss_norm = get_norm_regularization(scores) * weight
                    loss_norm = loss_norm * all_params['diversity_weight']

                    total_loss['div'] += loss_div.item()
                    total_loss['norm'] += loss_norm.item()

                out = out[:, :all_params['action_class_num']]  # Remove bg
                pred = torch.argmax(out, dim=1)
                correct += (pred.item() == label.item())

                ###############

                video_key = ''.join(video_name.split('-')
                                    [:-1])  # remove content after the last -

                pred_score_dict[video_key] = out.cpu().numpy()

                if video_key not in label_dict.keys():
                    label_dict[video_key] = np.zeros((1, all_params['action_class_num']))

                label_dict[video_key][0, label.item()] = 1
                ###############

        accuracy = correct / total_cnt
        total_loss[
            'sum'] = total_loss['cls'] + total_loss['div'] + total_loss['norm']
        avg_loss = {k: v / total_cnt for k, v in total_loss.items()}

        ##############
        pred_score_matrix = []
        label_matrix = []
        for k, v in pred_score_dict.items():
            pred_score_matrix.append(v)
            label_matrix.append(label_dict[k])

        return accuracy, avg_loss

    def train(train_train_loader, train_test_loader, test_test_loader, modality,
              naming, lr_val):
        plt.figure()
        #For plotting loss graph
        loss_val = []   
        step_list = []
        assert (modality in ['both', 'rgb', 'flow'])

        log_dir = os.path.join('logs', naming, modality)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        save_dir = os.path.join('models', naming)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if modality == 'both':
            model = BackboneNet(in_features=all_params['feature_dim'] * 2,
                                **all_params['model_params']).to(device)
        else:
            model = BackboneNet(in_features=all_params['feature_dim'],
                                **all_params['model_params']).to(device)

        optimizer = optim.Adam(model.parameters(),
                               lr=lr_val,
                               weight_decay=all_params['weight_decay'])

        if all_params['learning_rate_decay']:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[all_params['max_step_num'] // 2], gamma=0.1)

        optimizer.zero_grad()

        criterion = nn.CrossEntropyLoss(reduction='mean')

        update_step_idx = 0
        single_video_idx = 0
        loss_recorder = {
            'cls': 0,
            'div': 0,
            'norm': 0,
            'sum': 0,
        }

        while update_step_idx < all_params['max_step_num']:

            # Train loop
            for _, data in enumerate(train_train_loader):

                model.train()

                single_video_idx += 1
                label = data['label'].to(device)
                weight = data['weight'].to(device).float()
                model_input = None
                if modality == 'both':
                    if data['rgb'].shape[1] != 0 and data['flow'].shape[1] != 0:
                        rgb = data['rgb'].to(device)
                        flow = data['flow'].to(device)
                        model_input = torch.cat([rgb, flow], dim=2)
                    else: 
                        continue
                elif modality == 'rgb':
                    if data['rgb'].shape[1] != 0:
                        model_input = data['rgb'].to(device)
                    else:
                        continue
                else:
                    if data['flow'].shape[1] != 0:
                        model_input = data['flow'].to(device)
                    else:
                        continue
                #RGB or Flow
                if model_input.shape[-1] == all_params['feature_dim']:
                    model_input = model_input.transpose(2, 1)
                #Both
                if modality == 'both' and model_input.shape[-1] == all_params['feature_dim']*2:
                    model_input = model_input.transpose(2, 1)
                _, _, out, scores, _ = model(model_input) 

                loss_cls = criterion(out, label) * weight

                if all_params['diversity_reg']:
                    loss_div = get_diversity_loss(scores) * weight
                    loss_div = loss_div * all_params['diversity_weight']

                    loss_norm = get_norm_regularization(scores) * weight
                    loss_norm = loss_norm * all_params['diversity_weight']

                    loss = loss_cls + loss_div + loss_norm

                    loss_recorder['div'] += loss_div.item()
                    loss_recorder['norm'] += loss_norm.item()

                else:
                    loss = loss_cls

                loss_recorder['cls'] += loss_cls.item()
                loss_recorder['sum'] += loss.item()

                loss.backward()

                # Test and Update
                if single_video_idx % all_params['batch_size'] == 0:

                    # Test
                    if update_step_idx % all_params['log_freq'] == 0:

                        train_acc, train_loss = test(
                            model, train_test_loader, modality)
                        
                        if args.test_log:

                            test_acc, test_loss = test(
                                model, test_test_loader, modality)

                    print('Train Accuracy:{}, Test Accuracy:{}'.format(train_acc, test_acc))
                    
                    # Batch Update
                    update_step_idx += 1

                    for k, v in loss_recorder.items():

                        print('Step {}: Loss_{}-{}'.format(
                            update_step_idx, k, v / all_params['batch_size']))

                        #Plot loss over every iterations 
                        if k == 'sum':
                            step_list.append(update_step_idx)
                            loss_val.append(v/all_params['batch_size'])
                            plt.title('Classification accuracy: {:.4f}'.format(train_acc))
                            plt.xlabel('Iterations')
                            plt.ylabel('Loss')

                        loss_recorder[k] = 0
                    
                    if update_step_idx % all_params['log_freq'] == 0: #Plot graph every 500 iterations
                        plt.plot(step_list, loss_val)
                        img = 'Loss_LR-{}.png'.format(lr_val)
                        img_pth = os.path.join(log_dir, img)
                        plt.savefig(img_pth)

                    optimizer.step()
                    optimizer.zero_grad()

                    if all_params['learning_rate_decay']:
                        scheduler.step()

                    if update_step_idx in all_params['check_points']:
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                save_dir,
                                'model-{}-{}'.format(modality,
                                                     update_step_idx)))

                    if update_step_idx >= all_params['max_step_num']:
                        break

    
    train_dataset_dict = get_dataset(dataset_name=all_params['dataset_name'], 
                                     subset=args.train_subset_name, 
                                     file_paths=all_params['file_paths'],
                                     sample_rate=all_params['sample_rate'],
                                     base_sample_rate=all_params['base_sample_rate'],
                                     action_class_num=all_params['action_class_num'],
                                     modality='both',
                                     feature_type=all_params['feature_type'],
                                     feature_oversample=all_params['feature_oversample'],
                                     temporal_aug=True,
                                     load_background=all_params['with_bg'])

    train_train_dataset = SingleVideoDataset(train_dataset_dict,
                                             single_label=True,
                                             random_select=True,
                                             max_len=all_params['training_max_len'])  # To be checked

    train_test_dataset = SingleVideoDataset(train_dataset_dict,
                                            single_label=True,
                                            random_select=False,
                                            max_len=None)

    train_train_loader = torch.utils.data.DataLoader(train_train_dataset,
                                                     batch_size=1,
                                                     pin_memory=True,
                                                     shuffle=True)

    train_test_loader = torch.utils.data.DataLoader(train_test_dataset,
                                                    batch_size=1,
                                                    pin_memory=True,
                                                    shuffle=False)
    if args.test_log:
        test_dataset_dict = get_dataset(dataset_name=all_params['dataset_name'],
                                        subset=args.test_subset_name, 
                                        file_paths=all_params['file_paths'],
                                        sample_rate=all_params['sample_rate'],
                                        base_sample_rate=all_params['base_sample_rate'],
                                        action_class_num=all_params['action_class_num'],
                                        modality='both',
                                        feature_type=all_params['feature_type'],
                                        feature_oversample=all_params['feature_oversample'],
                                        temporal_aug=True,
                                        load_background=False)

        test_test_dataset = SingleVideoDataset(test_dataset_dict,
                                               single_label=True,
                                               random_select=False,
                                               max_len=None)

        test_test_loader = torch.utils.data.DataLoader(test_test_dataset,
                                                       batch_size=1,
                                                       pin_memory=True,
                                                       shuffle=False)
    else:

        test_test_loader = None

    #lrlog_range = [-3.4879, -3.0146]   
    for run_idx in range(all_params['train_run_num']):
        '''
        #Random search lr
        lr_log = np.random.uniform(low=lrlog_range[0], high=lrlog_range[1], size=None)
        lr_val = 10 ** lr_log
        '''
        lr_val=all_params['learning_rate']
        naming = '{}-run-{}'.format(all_params['experiment_naming'], run_idx)
        
        train(train_train_loader, train_test_loader, test_test_loader, 'rgb',
              naming, lr_val)
        
        train(train_train_loader, train_test_loader, test_test_loader, 'flow',
              naming, lr_val)
        
        train(train_train_loader, train_test_loader, test_test_loader, 'both',
              naming, lr_val)
