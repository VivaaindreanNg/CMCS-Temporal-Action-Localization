# CMCS-Temporal-Action-Localization

Code for 'Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization' (CVPR2019).

[Paper](http://www.vie.group/media/pdf/1273.pdf) and [Supplementary](http://www.vie.group/media/pdf/1273-supp.zip).

## Recommended Environment
* Python 3.5 and above
* Cuda 10.0
* PyTorch 0.4

## Prerequisites
* Install dependencies: `pip3 install -r requirements.txt`.
* Alternatively, setup the required environment based on [tal_env.py](https://github.com/VivaaindreanNg/CMCS-Temporal-Action-Localization/blob/master/tal_env.yml). (Change the value of 'name' and 'prefix' to suit your needs)
* [Install Matlab API for Python](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) (matlab.engine).
* Prepare UCF-Crime datasets.

### Feature Extraction

We employ I3D features in the paper. 
[Head over to here for more info.](https://github.com/VivaaindreanNg/CMCS-Temporal-Action-Localization/tree/master/pytorch-i3d-feature-extraction) Other features can also be used.

### Generate Static Clip Masks:

Static clip masks are used for hard negative mining. They are included in the download features.
If you want to generate the masks by yourself, please refer to `tools/get_static_clips.py`.

### Data preparation for Training & Testing

Run the `tools/train_test_split.py` to segregate extracted features into val(training subdirectories) & test(testing subdirectories), similar as the one configured in `configs/ucf_crime-I3D.json`. 

## Run

1. Train models with weak supervision (Skip this if you use our trained model):
```
python3 train.py --config-file {} --train-subset-name {} --test-subset-name {} --test-log
```

2. Test and save the class activation sequences (CAS):
```
python3 test.py --config-file {} --train-subset-name {} --test-subset-name {} --no-include-train
```

3. Action localization using the CAS:
```
python3 detect.py --config-file {} --train-subset-name {} --test-subset-name {} --no-include-train
```

For THUMOS14, predictions are saved in `output/predictions` and final performances are saved in a npz file in `output`.
For ActivityNet, predictions are saved in `output/predictions` and final performances can be obtained via the dataset evaluation API.

#### Settings
This experiment is evaluated on UCF-Crime with I3D features. Experiment settings and their auguments are listed as following. 

|           config-file          | train-subset-name | test-subset-name |
|:------------------------------:|:-----------------:|:----------------:|
|     configs/ucf_crime-I3D.json |        val        |       test       |


## Trained Models

Our trained models are provided [in this folder](https://github.com/Finspire13/Weakly-Action-Detection/tree/Release-CVPR19/models). To use these trained models, run `test.py` and `detect.py` with the config files [in this folder](https://github.com/Finspire13/Weakly-Action-Detection/tree/Release-CVPR19/configs/trained).

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

