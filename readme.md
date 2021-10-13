# 3D MOT on Waymo dataset
This project is developed for online 3D multi-object tracking on Waymo dataset. It is build upon Cascade3D
 detector and CA-based KF tracker. The tracking code
 is from [here](https://github.com/hailanyi/3D-Multi-Object-Tracker); detector is from [here](https://github.com/hailanyi/Cascade3D); visualization code is from
[here](https://github.com/hailanyi/3D-Detection-Tracking-Viewer).
![](./doc/demo.gif)

## Waymo Results
**Vehicle/Pedestrian/Cyclist** online tracking results, evaluated by **MOTA/L2** metrics of Waymo benchmark. 

|set|Vehicle_L2|Pedestrian_L2|Cyclist_L2| ALL_NS_L2 |
|:---:|:---:|:---:|:---:|:---:|
|val set|59.30|62.78|61.72|61.27|
|test set|63.66|64.79|59.34|62.60|
 
## Prepare data 
You can download the waymo dataset infos and Cascade3D detections from google drive.
 To obtain the download link, please send us (wuhai@stu.xmu.edu.cn) an email  with your name, institute, a screenshot of the Waymo dataset download page 
(please note that Waymo open dataset is under strict non-commercial license).

## Quick start
* Please modify the info path, detections path in the [yaml file](./config/config.yaml) 
to your own path.
* Then run ``` python3 waymo_tracking.py config/config.yaml``` 
* The results are automatically saved to ```save_path``` in yaml file, and 
evaluated by Waymo metrics (the results are lower than the one calculated by Waymo benchmark).
You can create a submission file to evaluate the performance on Waymo benchmark by running ``` python3 create_submission/create_submission.py```

## Citation
@inproceedings{sun2020scalability, 
title={Scalability in perception for autonomous driving: Waymo open dataset}, 
author={Sun, Pei and Kretzschmar, Henrik and Dotiwalla, Xerxes and Chouard, Aurelien and Patnaik, Vijaysai and Tsui, Paul and Guo, James and Zhou, Yin and Chai, Yuning and Caine, Benjamin and others}, 
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
 pages={2446--2454}, year={2020} }
 
@article{wu20213d,
title={3D Multi-Object Tracking in Point Clouds
Based on Prediction Confidence-Guided Data
Association},
author={Wu, Hai and Han, Wenkai and Wen, Chenglu
and Li, Xin and Wang, Cheng},
journal={IEEE TITS},
year={2021}
}

