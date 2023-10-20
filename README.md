# robotics datasets

* curation effort to document and consolidate as exhaustive as possible a list of datasets that may be useful for robotics research. 
* motivation:
    - deep learning/llms for robotics is fundamentally data hungry
    - lots of distributed datasets shared generously by different robotics labs/ institutions
        - not many efforts at consolidation 
    - multiple heterogenous datasets for different hardware, settings, environment
    - make it easier to inspect different robotics datasets for own custom problem

# awesome-robotics-datasets github list
- https://github.com/mint-lab/awesome-robotics-datasets/blob/master/README.md


## Dataset Collections
* **Robotics**
  * ~~[Radish: The Robotics Data Set Repository](http://radish.sourceforge.net/), Andrew Howard and Nicholas Roy~~ (Not working)
  * [Repository of Robotics and Computer Vision Datasets](https://www.mrpt.org/robotics_datasets), MRPT
    * :memo: It includes _Malaga datasets_ and some of classic datasets published in [Radish](http://radish.sourceforge.net/).
  * [IJRR Data Papers](http://journals.sagepub.com/topic/collections/ijr-3-datapapers/ijr), IJRR
  * [Awesome SLAM Datasets](https://github.com/youngguncho/awesome-slam-datasets), Younggun Cho :+1:
* **Computer Vision**
  * [CVonline Image Databases](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm), CVonline
  * [Computer Vision Datasets on the Web](http://www.cvpapers.com/datasets.html), CVPapers :+1:
  * [YACVID: Yet Another Computer Vision Index To Datasets](http://riemenschneider.hayko.at/vision/dataset/), Hayko Riemenschneider :+1:
  * [Computer Vision Online Datasets](https://computervisiononline.com/datasets), Computer Vision Online
* **Others**
  * [Machine Learning Repository](http://archive.ics.uci.edu/ml), UCI
  * [Kaggle Datasets](https://www.kaggle.com/datasets), Kaggle
  * [IEEE DataPort](https://ieee-dataport.org/), IEEE


## Place-specific Datasets
### Driving Datasets
* [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/) and [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/), Andreas Geiger et al. :+1:
  * [SemanticKITTI](http://semantic-kitti.org/), Jens Behley et al.
* [Waymo Open Dataset](https://waymo.com/open), Waymo
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
* [AppoloScape Dataset](http://apolloscape.auto/)
* [Berkely DeepDrive Dataset](https://bdd-data.berkeley.edu/) (BDD100K), BAIR at UC Berkely
* [nuScenes Dataset](https://www.nuscenes.org/), APTIV
* [$D^2$-City Dataset](https://outreach.didichuxing.com/d2city/d2city), DiDi
* [Ford Campus Vision and Lidar Data Set](http://robots.engin.umich.edu/SoftwareData/Ford), PeRL at Univ. of Michigan
* [MIT DARPA Urban Challenge Dataset](http://grandchallenge.mit.edu/wiki/index.php?title=PublicData), MIT
* [KAIST Multi-spectral Recognition Dataset in Day and Night](https://sites.google.com/view/multispectral/), RCV Lab at KAIST
* [KAIST Complex Urban Dataset](http://irap.kaist.ac.kr/dataset/), IRAP Lab at KAIST
* [New College Dataset](http://www.robots.ox.ac.uk/NewCollegeData/index.php), MRG at Oxford Univ.
* [Chinese Driving from a Bike View](http://www.sujingwang.name/CDBV.html) (CDBV), CAS
* [CULane Dataset](https://xingangpan.github.io/projects/CULane.html), CUHK
* [ROMA (ROad MArkings) Image Database](http://perso.lcpc.fr/tarel.jean-philippe/bdd/), Jean-Philippe Tarel et al.

### Flying Datasets
* [The Zurich Urban Micro Aerial Vehicle Dataset](http://rpg.ifi.uzh.ch/zurichmavdataset.html), RPG at ETHZ
* [The UZH-FPV Drone Racing Dataset](http://rpg.ifi.uzh.ch/uzh-fpv.html), RPG at ETHZ
* [MultiDrone Public Dataset](https://multidrone.eu/multidrone-public-dataset/), MultiDrone Project
* [The Blackbird Dataset](https://github.com/mit-fast/Blackbird-Dataset), AgileDrones Group at MIT

### Underwater Datasets
* [Marine Robotics Datasets](http://marine.acfr.usyd.edu.au/datasets/), ACFR

### Outdoor Datasets
* [The Rawseeds Project](http://www.rawseeds.org/)
  * :memo: It includes _Bovisa_ dataset is for outdoor and _Bicocca_ dataset is for indoor.
* [Planetary Mapping and Navigation Datasets](http://asrl.utias.utoronto.ca/datasets/), ASRL at Univ. of Toronto

### Indoor Datasets
* [Robotics 2D-Laser Datasets](http://www.ipb.uni-bonn.de/datasets/), Cyrill Stachniss
  * :memo: It includes some of classic datasets published in [Radish](http://radish.sourceforge.net/).
* [Long-Term Mobile Robot Operations](http://robotics.researchdata.lncn.eu/), Lincoln Univ.
* [MIT Stata Center Data Set](http://projects.csail.mit.edu/stata/), Marine Robotics Group at MIT
* [KTH and COLD Database](https://www.pronobis.pro/#data), Andrzej Pronobis
* [Shopping Mall Datasets](http://www.irc.atr.jp/sets/TEMPOSAN_dataset/), IRC at ATR
* [RGB-D Dataset 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/), Microsoft


## Topic-specific Datasets for Robotics
### Localization, Mapping, and SLAM
* [SLAM Benchmarking](http://ais.informatik.uni-freiburg.de/slamevaluation/), AIS at Univ. of Freiburg
* [Robotic 3D Scan Repository](http://kos.informatik.uni-osnabrueck.de/3Dscans/), Univ. of Wurzburg and Univ. of Osnabruck
* [3D Pose Graph Optimization](https://lucacarlone.mit.edu/datasets/), Luca Carlone
* **Landmark-based Localization**
  * [Range-only Data for Localization](http://www.frc.ri.cmu.edu/projects/emergencyresponse/RangeData/), CMU RI
  * [Roh's Angulation Dataset](https://github.com/sunglok/TriangulationToolbox/tree/master/dataset_roh), HyunChul Roh
  * [Wireless Sensor Network Dataset](http://www.cs.virginia.edu/~whitehouse/research/localization/), Kamin Whitehouse

### Path Planning and Navigation
* [Pathfinding Benchmarks](http://www.movingai.com/benchmarks/), Moving AI Lab at Univ. of Denver
* [Task and Motion Planner Benchmarking](http://www.neil.dantam.name/2018/rss-tmp-workshop/#benchmarks), RSS 2018 Workshop


## Topic-specific Datasets for Computer Vision
### Features
* [Affine Covariant Features Datasets](https://www.robots.ox.ac.uk/~vgg/data/affine/), VGG at Oxford
  * [Repeatability Benchmark Tutorial](https://www.vlfeat.org/benchmarks/overview/repeatability.html), VLFeat
* [A list of feature performance evaluation datasets](https://github.com/openMVG/Features_Repeatability), maintained by openMVG

### Saliency and Foreground
* **Saliency**
  * [MIT Saliency Benchmark](http://saliency.mit.edu/), MIT
  * [Salient Object Detection: A Benchmark](http://mmcheng.net/salobjbenchmark/), Ming-Ming Cheng
* **Foreground/Change Detection (Background Subtraction)**
  * [ChangeDetection.NET](http://www.changedetection.net/) (a.k.a. CDNET)

### Motion and Pose Estimation
* [AdelaideRMF: Robust Model Fitting Data Set](https://cs.adelaide.edu.au/~hwong/doku.php?id=data), Hoi Sim Wong

### Structure-from-Motion and 3D Reconstruction
* **Objects**
  * [IVL-SYNTHESFM v2](https://board.unimib.it/datasets/fnxy8z8894/1), Davide Marelli et al.
  * [Fuji-SfM Dataset](https://zenodo.org/record/3712808#.YSfTs44zaUl), Jordi Gene-Mola et al.
  * [Large Geometric Models Archive](https://www.cc.gatech.edu/projects/large_models/), Georgia Tech
  * [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/), Stanford Univ.
* **Places**
  * [Photo Tourism Data](http://phototour.cs.washington.edu/), UW and Microsoft

### Object Tracking
* [Visual Object Tracking Challenge](http://www.votchallenge.net/) (a.k.a. VOT) :+1:
* [Visual Tracker Benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/) (a.k.a. OTB)

### Object, Place, and Event Recognition
* **Pedestrians**
  * [EuroCity Persons Dataset](https://eurocity-dataset.tudelft.nl/) (a.k.a. ECP)
  * [Daimler Pedestrian Benchmark Data Sets](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html)
  * [CrowdHuman](http://www.crowdhuman.org/)
* **Objects**
  * [RGB-D Object Dataset](http://rgbd-dataset.cs.washington.edu/), UW
  * [Sweet Pepper and Peduncle 3D Datasets](http://enddl22.net/wordpress/datasets/sweet-pepper-and-peduncle-3d-datasets), InKyu Sa
* **Places**
  * [Loop Closure Detection](http://cogrob.ensta-paristech.fr/loopclosure.html), David Filliat et. al.
* **Traffic and Surveillance**
  * [BEST: Benchmark and Evaluation of Surveillance Task](http://best.sjtu.edu.cn/Data/List/Datasets), SJTU
  * [VIRAT Video Dataset](http://www.viratdata.org/)


## Research Groups
* [TUM CVG Datasets](https://vision.in.tum.de/data/datasets)
  * Tags: Visual(-inertia) odometry, visual SLAM, 3D reconstruction
* [Oxford VGG Datasets](http://www.robots.ox.ac.uk/~vgg/data/)
  * Tags: Visual features, visual recognition, 3D reconstruction
* [QUT CyPhy Datasets](https://wiki.qut.edu.au/display/cyphy/Datasets)
  * Tags: Visual SLAM, LiDAR SLAM
* [Univ. of Bonn Univ. Stachniss Lab Datasets](https://www.ipb.uni-bonn.de/data/)
  * Tags: SLAM
* [EPFL CVLAB Datasets](https://cvlab.epfl.ch/data)
  * Tags: 3D reconstruction, local keypoint, optical flow, RGB-D pedestrian
* [The Middlebury Computer Vision Pages](http://vision.middlebury.edu/)
  * Tags: Stereo matching, 3D reconstruction, MRF, optical flow, color
* [Caltech CVG Datasets](http://www.vision.caltech.edu/archive.html)
  * Tags: Objects (pedestrian, car, face), 3D reconstruction (on turntables)


# DLR 

* [Scientific datasets list](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-13772/)

## datasets

* [Benchmark Maps](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-18664/)
    * 2.5D elevation maps of planetary environment that were collected on Mt. Etna during the space-analogous ARCHES mission. In addition to the raw elevation maps, we provide cost maps that encode the traversibility of the terrain.
	
[CROOS-CV](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-13773/)
The CROOS-CV dataset is intended to support and benchmark Computer Vision (CV) development for Close Range On-Orbit Servicing (CROOS). It is an representative image dataset for CROOS operations with distances of 2 m between servicer and client satellite that was recorded under illumination conditions similar to a Low Earth Orbit.
	
[HOWS-CL-25](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-18900/)
HOWS-CL-25 is a synthetic dataset especially designed for object classification on mobile robots operating in a changing environment (like a household), where it is important to learn new, never seen objects on the fly.
	
[Long Range Navigation Tests (LRNTs)](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-13819)
During the ROBEX demo mission space campaign that took place during June–July 2017 on Mt. Etna, Italy, we performed some Long Range Navigation Tests.
	
[Morocco Navigation Tests](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-13824)
The Institute was part of the PERASPERA Space Robotics Technologies Cluster in the operational grants OG3 and OG6. In that context, the research group participated in the 2018 November/December field test in the Moroccan desert close to the city of Erfoud.
	
[MMX Navigation Testing Data Set](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-18676/)
Public collection of test data that is used to test the DLR Autonomous Navigation Experiment on the MMX Rover for Phobos
	
[Planetary Stereo Solid-State LiDAR Inertial Dataset](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-18306)
e release a dataset recorded on the Moon-like environment of Mount Etna, Sicily, with a sensor setup that comprises a stereo camera, a LiDAR and an IMU.
	
[ReSyRIS](https://www.dlr.de/rm/en/desktopdefault.aspx/tabid-18801/)
The Real-Synthetic Rock Instance Segmentation dataset (ReSyRIS) is created for training and evaluation of rock segmentation, detection and instance segmentation in (quasi-)extra-terrestrial environments.
	
STIOS
The Stereo Instances on Surfaces Datensatz (STIOS) is created for evaluation of instance-based algorithm and mainly intended for robotic applications, which is why the dataset refers to horizontal surfaces.
thr_kr16_120 	
THR Dataset
The THR (Top Hat Rail) data set consists of color and depth images from different objects taken from multiple views in different scenes. The data set consists of 9 object classes and can be used e.g. to improve perception algorithms by learning. 


    - Apollo Scape
        http://apolloscape.auto/scene.html

    - Agarwal etc. al (2020): Ford Multi-AV Seasonal Dataset. [arXiv:2003.07969](https://arxiv.org/abs/2003.07969), [Ford AV Datasets](https://avdata.ford.com/)
    
    - Behley et al. (2019): SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences. arXiv:1904.01416
    - Bosch Small Traffic Lights Dataset
        Object detection dataset for traffic lights in road recordings

    Braun et al. (2019): The EuroCity Persons Dataset. link
    Caesar et al. (2019): nuScenes: A multimodal dataset for autonomous driving. arXiv:1903.11027; nuScenes includes RADAR data as well
    CARLA Imitation learning
    Change et al. (2019): Argoverse: 3D Tracking and Forecasting with Rich Maps. CVPR 2019.PDF.
    Chen et al. (2016): Anticipating Accidents in Dashcam Videos. ACCV 2016 Oral.
    Chilean Underground Mine Dataset

    Cityscapes

    Déziel et al. (2021): PixSet : An Opportunity for 3D Computer Vision to Go Beyond Point Clouds With a Full-Waveform LiDAR Dataset. arXiv: 2102.12010, Leddar PixSet

    Geyer et al. (2020): A2D2: Audi Autonomous Driving Dataset. arXiv: 2004.06320; A2D2
    Kesten et al. (2019): Lyft Level 5 AV Dataset 2019

    KITTI Vision Benchmark Suite
    Maddern et al. (2016): 1 Year, 1000km: The Oxford RobotCar Dataset. The International Journal of Robotics Research (IJRR), 2016. pdf
    Mallios et al. (2017): Underwater caves sonar and vision data set. The International Journal of Robotics Research, 2017 (36), 1247-1251. doi: 10.1177/0278364917732838

    Mapillary Vistas

    PandaSet by Hesai and Scale AI

    Schafer et al. (2018): A Commute in Data: The comma2k19 Dataset. arXiv:1812.05752; comma2k19

    Udacity Self-Driving Car Dataset; relabeled on roboflow

    Waymo dataset
    Yu et al. (2018): BDD100K: A Diverse Driving Video Database with Scalable Annotation Tooling. arXiv:1805.04687; Berkeley DeepDrive

