# CMCS-Temporal-Action-Localization

Code for 'Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization' (CVPR2019).

[Paper](http://www.vie.group/media/pdf/1273.pdf) and [Supplementary](http://www.vie.group/media/pdf/1273-supp.zip).

## Recommended Environment
* Python 3.5 and above
* Cuda 10.0
* PyTorch 0.4 or 1.2.0

## Prerequisites
* Install dependencies: `pip3 install -r requirements.txt`.
* Alternatively, setup the required environment based on [tal_env.py](https://github.com/VivaaindreanNg/CMCS-Temporal-Action-Localization/blob/master/tal_env.yml). (Change the value of 'name' and 'prefix' to suit your needs)
* [Install Matlab API for Python](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) (matlab.engine).
* Prepare UCF-Crime datasets.

### Feature Extraction

We employ I3D features in the paper. 
[Head over to here for more info.](https://github.com/VivaaindreanNg/CMCS-Temporal-Action-Localization/tree/master/pytorch-i3d-feature-extraction) Other features can also be used.

### Generate Static Clip Masks:

Static clip masks are used for hard negative mining. 
If you want to generate the masks by yourself, please refer to `tools/get_static_clips.py`.

### Data preparation for Training & Testing

Run the `tools/train_test_split.py` to segregate extracted features into val(training subdirectories) & test(testing subdirectories), similar as the one configured in `configs/ucf_crime-I3D.json`. 

## Run (All train, test and evaluation were done with pytorch==1.2.0)

1. Train models with weak supervision via torch-nightly==1.2.0 (Skip this if you use our trained model):
```
python train.py --config-file {} --train-subset-name val --test-subset-name test --test-log
```

2. Test and save the class activation sequences (CAS):
```
python test.py --config-file {} --test-subset-name {}
```

3. Compute area under ROC (at frame-level) using the saved CAS:
```
python evaluate_roc.py --config-file {} --test-subset-name {}
```

4. Output localized actions & calculate mAP:
```
python evaluate_map.py --config-file {} --test-subset-name {}
```
For UCF-Crime, localized predictions (for each modalities) are saved in `outputs/predictions` and final performances (mAP for each modalities spanning across .1 to .5 IoU thresholds) are saved in a json file in `outputs`.

#### Settings
This experiment is evaluated on UCF-Crime with I3D features. Experiment settings and their arguments are listed as following. 

|           config-file          | train-subset-name | test-subset-name |
|:------------------------------:|:-----------------:|:----------------:|
|     configs/ucf_crime-I3D.json |        val        |       test       |


## Trained Models

Trained models are provided [in this folder](https://github.com/VivaaindreanNg/CMCS-Temporal-Action-Localization/tree/master/models/ucf_crime-I3D-run-0). To use these trained models, run `test.py` with the config file [in this folder](https://github.com/VivaaindreanNg/CMCS-Temporal-Action-Localization/tree/master/configs).

## Citation
@InProceedings{Liu_2019_CVPR,
author = {Liu, Daochang and Jiang, Tingting and Wang, Yizhou},
title = {Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}

## License
MIT

