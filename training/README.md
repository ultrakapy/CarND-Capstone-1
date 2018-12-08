# CarND-Capstone-Training
### 1.Images(2018-12-01):
#### SimulatorTrack1_Classified_Imgs
RED: 706   
YELLOW: 66   
GREEN: 489  
UNKNOWN: 288   
#### RealCarTrack_Unclassified_Imgs
243  
#### FromNet_Traffic_Light_Imgs , images from UdaCity "Intro to Self-Driving Cars" course
training: 1187  
test: 297   
   
   
   
### 2. a bit of py scripts(2018-12-03) 
#### Functions in train.py
get_traindata() # read image from disk into 2 python lists.  
train() # a cnn to test data   
   
need:  
tensorflow 1.3.0  
sklearn  

usage:  
```
python train.py  
```
   
result:  
...  
112  train_accuracy:  0.78   
113  train_accuracy:  0.74  
114  train_accuracy:  0.76  
115  train_accuracy:  0.78  
116  train_accuracy:  0.86  
117  train_accuracy:  0.84  
118  train_accuracy:  0.82  
119  train_accuracy:  0.88  
120  train_accuracy:  0.82  
121 -----Test_accuracy:  [0.85483873]  
   
   
### 3. Manual annotation(2018-12-03)
####  Add 60 simulator image annotations manually (use labelimg,  20 XMLs per traffic light color) for test, if test is ok, label image else. 
   
label file folder:   
./SimulatorTrack1_Classified_Imgs/GREEN_label   
./SimulatorTrack1_Classified_Imgs/RED_label   
./SimulatorTrack1_Classified_Imgs/YELLOW_label   
   
XML example (NOTE: 4th line in XML has a full path, when use those files, maybe need to modify it):    
```
<annotation>
        <folder>RED</folder>
        <filename>IMG_15437218315.jpg</filename>
        <path>/jixj/term3/p015/SimulatorTrack1_Classified_Imgs/RED/IMG_15437218315.jpg</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>800</width>
                <height>600</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
                <name>red</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>126</xmin>
                        <ymin>256</ymin>
                        <xmax>181</xmax>
                        <ymax>382</ymax>
                </bndbox>
        </object>
        <object>
                <name>red</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>374</xmin>
                        <ymin>259</ymin>
                        <xmax>432</xmax>
                        <ymax>386</ymax>
                </bndbox>
        </object>
        <object>
                <name>red</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>624</xmin>
                        <ymin>263</ymin>
                        <xmax>688</xmax>
                        <ymax>390</ymax>
                </bndbox>
        </object>
</annotation>

```
predefined_classes.txt is a classes file for labelimg. 

   
  


### 4. Train a ssd model(2018-12-05)
#### Navigate to CarND-Capstone/training
NOTICE: change below command '/jixj/term3/p021' to fullpath of CarND-Capstone/training  and change ssd_inception_v2_coco.config line 152, 172, 174, 186, 188 path to fullpath of CarND-Capstone/training   
    
#### If Tensorflow dose not install yet, Install TensorFlow version 1.3 by executing
```
pip install tensorflow==1.3
```
#### Install the following packages
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
sudo pip2 install  Cython
sudo pip2 install  contextlib2
sudo pip2 install  jupyter
sudo pip2 install  matplotlib
sudo pip2 install  absl-py
sudo pip2 install  pycocotools
sudo pip2 install  sklearn
sudo pip2 install  pandas==0.22.0

```
#### Clone TensorFlow's models repository from the tensorflow directory by executing
```
git clone https://github.com/tensorflow/models.git
```
#### Navigate to the models directory in the Command Prompt and execute
```
cd models

git checkout f7e99c0

```
#### Navigate to the ./research folder and execute

```
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python setup.py build
sudo python setup.py install

```
#### Test environment. If no error, environment is ok
```
python object_detection/builders/model_builder_test.py

python ./object_detection/exporter_test.py

```

#### Navigate to the CarND-Capstone/training folder
```
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
tar zxvf ssd_inception_v2_coco_11_06_2017.tar.gz

# delete all files in CarND-Capstone/training/model, CarND-Capstone/training/train, CarND-Capstone/training/test

python xml_to_csv.py 'SimulatorTrack1_Classified_Imgs'
python generate_tfrecord.py --csv_input=./train/SimulatorTrack1_Classified_Imgs_train.csv --output_path=./train/SimulatorTrack1_Classified_Imgs_train.record
python generate_tfrecord.py --csv_input=./test/SimulatorTrack1_Classified_Imgs_test.csv --output_path=./test/SimulatorTrack1_Classified_Imgs_test.record



```
#### Navigate to models/research 
```

python ./object_detection/train.py --logtostderr --train_dir=/jixj/term3/p021/model --pipeline_config_path=/jixj/term3/p021/ssd_inception_v2_coco.config


#result like:
#reference connection and already has a device field set to /device:CPU:0
#INFO:tensorflow:Starting Session.
#INFO:tensorflow:Saving checkpoint to path /jixj/term3/p021/model/model.ckpt
#INFO:tensorflow:Starting Queues.
#INFO:tensorflow:global_step/sec: 0
#INFO:tensorflow:Recording summary at step 0.
#INFO:tensorflow:global step 1: loss = 25.6500 (17.872 sec/step)
#INFO:tensorflow:global step 2: loss = 26.4797 (7.996 sec/step)
#INFO:tensorflow:global step 3: loss = 22.3657 (7.960 sec/step)
#INFO:tensorflow:global step 4: loss = 20.3114 (8.018 sec/step)
#INFO:tensorflow:global step 5: loss = 20.1324 (7.970 sec/step)
#INFO:tensorflow:Stopping Training.
#INFO:tensorflow:Finished training! Saving model to disk.

```


#### Navigate to the models directory
```
git checkout 9a811d95c478b062393debd1559df61490b97149

```

#### Navigate to models/research 
```
python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /jixj/term3/p021/ssd_inception_v2_coco.config --trained_checkpoint_prefix  /jixj/term3/p021/model/model.ckpt-5 --output_directory  /jixj/term3/p021/model

#result like:
#2018-12-04 22:32:02.758329: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
#2018-12-04 22:32:02.758380: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
#2018-12-04 22:32:02.758389: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
#2018-12-04 22:32:02.758393: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
#2018-12-04 22:32:02.758398: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
#Converted 410 variables to const ops.


```


Then it will get a frozen_inference_graph.pb in CarND-Capstone/training/model/
   
  
### 5. Use the trained model to predict test image (2018-12-05)
NOTICE:change /jixj/term3/p028/training to your CarND-Capstone/training
#### Navigate to the CarND-Capstone/training/utils folder
```
# delete all files in CarND-Capstone/training/utils/output
python img_predict.py --model_name=/jixj/term3/p028/training/model/ --path_to_label=/jixj/term3/p028/training/labelmap.pbtxt --test_image_path=test


#result like:
#1 boxes in /jixj/term3/p028/training/utils/test/IMG_154372566237.jpg image tile!
#('/jixj/term3/p028/training/utils/test/IMG_154372566237.jpg', '-->', 'output/IMG_154372566237.png')
#1 boxes in /jixj/term3/p028/training/utils/test/IMG_154372184947.jpg image tile!
#('/jixj/term3/p028/training/utils/test/IMG_154372184947.jpg', '-->', 'output/IMG_154372184947.png')
#1 boxes in /jixj/term3/p028/training/utils/test/IMG_15437218315.jpg image tile!
#('/jixj/term3/p028/training/utils/test/IMG_15437218315.jpg', '-->', 'output/IMG_15437218315.png')
#1 boxes in /jixj/term3/p028/training/utils/test/IMG_154372567036.jpg image tile!
#('/jixj/term3/p028/training/utils/test/IMG_154372567036.jpg', '-->', 'output/IMG_154372567036.png')
#1 boxes in /jixj/term3/p028/training/utils/test/IMG_154372544995.jpg image tile!
#('/jixj/term3/p028/training/utils/test/IMG_154372544995.jpg', '-->', 'output/IMG_154372544995.png')
#1 boxes in /jixj/term3/p028/training/utils/test/IMG_154372186749.jpg image tile!
#('/jixj/term3/p028/training/utils/test/IMG_154372186749.jpg', '-->', 'output/IMG_154372186749.png')



```
  
  
  

### Using rosbag, get real track images (2018-12-08)
   
#### 1. Turn on the save switch
```
# modefy this code 
SAVE_TRAFFIC_LIGHT_IMG = False
# to
SAVE_TRAFFIC_LIGHT_IMG = True

```
   
#### 2. Uncomment code
```
# in ros/src/tl_detector/tl_detector.py  function image_cb(), uncomment below code.   

# self.save_img(msg, 4)
```
   
#### 3. Start up rosbag.
see [link](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/3251f513-2f82-4d5d-88b6-9d646bbd9101)

   
#### 4. image position
ros/src/tl_detector/light_classification/IMGS/UNKNOWN
  
  
  

### Added labeled image monitor (2018-12-08)
#### 1. in ros/src/tl_detector/light_classification/tl_classifier.py , enalbe SHOW_MONITOR_IMAGE and "import"s
```
# to show labeled image, for test only, if submit code to udacity, should be set SHOW_MONITOR_IMAGE = False and comment below "imort" lines.  
# take a show: open a new terminal ,and run "rosrun image_view image_view image:=/clssifier_monitor_image"
SHOW_MONITOR_IMAGE = True
import visualization_utils as vis_util
import rospy
from sensor_msgs.msg import Image as Image_msg
from cv_bridge import CvBridge
```
   
#### 2. open a new terminal ,and run "rosrun image_view image_view image:=/clssifier_monitor_image"

