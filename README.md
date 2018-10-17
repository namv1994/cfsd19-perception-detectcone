# cfsd18-perception-detectcone

[![Build Status](https://travis-ci.org/cfsd/cfsd18-perception-detectcone.svg?branch=master)](https://travis-ci.org/cfsd/cfsd18-perception-detectcone)

### How it works
This microservice reads image from shared memory and extract RoI (region of interest, i.e. cone position) from image. Either ORB or Lidar+ORB is applied to find RoI. ORB (an OpenCV keypoint detector) detects keypoints/cones in the image and return back the 2D positions which are later triangulated to 3D ones. The 3D points are filtered, and then clustered via k-d tree. Median point of each cluster/group is computed, representing position of a cone. 3D points are project back to 2D positions, and we can extract RoI which is centered at the corresponding median point. However, with Lidar+ORB, lidar RoI is matched to ORB RoI, providing higher accuracy of position. With either one of the two methods, we can obtain RoI and do classification on it with CNN model. At the end, this microservice will send out messages including cone position and cone color.


### OD4Session message in and out
- recieve:
  - opendlv.logic.perception.ObjectDirection
  - opendlv.logic.perception.ObjectDistance
    - sender: (sensation-attention, 116)
  - opendlv.proxy.SwitchStateReading
    - sender: (?, 1401)
- send:
  - opendlv.system.SignalStatusMessage
  - opendlv.logic.perception.ObjectDirection
  - opendlv.logic.perception.ObjectDistance
  - opendlv.logic.perception.ObjectType
  - opendlv.logic.perception.Object
    - sender: (perception-detectcone, 118)


### Command line arguments
| parameter | comments |
| ----- | ----- |
| cid | OpenDaVINCI v4 session identifier [1..254] |
| name | name of the shared memory area; must start with '/' and be no longer than 255, if not it will be adjusted accordingly |
| width | image width, image is from camera |
| height | image height |
| bpp | bits per pixel, should be either 24 or 8 |
| threshold | if the predicted probability is bigger than the threshold, then the corresponding cone can be labeled |
| timeDiffMilliseconds | time limit of collecting a specific number of lidar cones |
| separationTimeMs | time limit of collecting a lidar cone in CollectCones function |
| checkLidarMilliseconds | time limit of recieving lidar data |
| senderStamp | sender stamp of this microservice (118) |
| attentionSenderStamp | sender stamp of sensation-attention (116) |
| offline | if true, will not save data from shared memory as png images, but will use the already existed png images for further processing |
| annotate | if ture, make annotation |
| stateMachineId | opendlv.proxy.SwitchStateReading envelope's senderstamp |
| readyStateMachine | if ture, will set the drivingState true |
| forwardDetection | if ture, will call forwardDetectionORB |
| fastThreshold | size of the border where the features are not detected, set by ORB detector in backwardDetection |
| matchDistance | [not found usage in code] |
- example usage:

    cfsd18-perception-detectcone --cid=${CID} --name=cam0 --width=2560 --height=720 --bpp=24 --threshold=0.6 --timeDiffMilliseconds=40 --separationTimeMs=20 --checkLidarMilliseconds=1000 --senderStamp=118 --attentionSenderStamp=116 --offline=0 --annotate=0 --stateMachineId=1401 --readyStateMachine=0 --forwardDetection=0 --fastThreshold=20 --orbPatchSize=31 --matchDistance=1.5


### Function call graph
- main()
  - detectcone.setTimeStamp(std::pair<int64_t, cv::Mat>)  [pass timestamp and image to detectcone]
  - detectcone.checkLidarState()
    - **detectcone.forwardDetectionORB(cv::Mat image)**
  - od4.dataTrigger -> envelopeRecieved  [recieve message sent by sensation-attention]
    - collector.CollectorCones(cluon::data::Envelope)
    - [independent thread] collector.InitializeCollection()
      - collector.SendFrame()
        - detectcone.recieveCombinedMessage(cluon::data::TimeStamp, std::map<int,ConePackage>)
          - detectcone.SendCollectorCones(Eigen::MatrixXd)
            - **detectcone.backwardDetection(cv::Mat image, Eigen::MatrixXd lidarCones, int64_t minValue)**


### main function
- parse command line arguments
- create folders and files:
  - /opt/yyyy-mm-dd-hh-mm-ss/  (e.g. 2018-07-13-14-58-45)
  - /opt/yyyy-mm-dd-hh-mm-ss/timestamp/timestamps.txt
  - /opt/yyyy-mm-dd-hh-mm-ss/images/n.png  (n = 0,1,2,...)
  - /opt/yyyy-mm-dd-hh-mm-ss/results/log.txt
- attach to shared memory, create image header and read data from shared memory
- if in drivingState, call function detectcone.checkLidarState; and save frame image as n.png (n = 0,1,2,...)
- register lambda function for handling recieved message, if recieve cone direction or distance, call function collector.CollectCones
- send out opendlv.system.SignalStatusMessage


### forwardDetectionORB: ORB + CNN
- take image frame as input
- create an OpenCV ORB detector, and set nfeatures = 100 (maximum number of featrues to retain)
- ORB detects keypoints within a specific row range of the image
- 2D position from keypoints, and triangulate these points to 3D world
- discard the 3D keypoints that are outside the effective region, and then filter the rest 3D keypoints further
- group 3D points by kd-tree, compute median point for each group
- project back the 3D points to 2D points, and extract RoI (cv::Rect) which is centered at the median point
- patch the image with the RoI
- convert the patched image to CNN inputs
- given the inputs, predict the probability of each color with pre-trained CNN model
- find each cone's maxProb and the corresponding position
- if annotate is true, make annotation
- save the processed image with results as n.png (n = 0,1,2,...) via an independent thread
- call SendMatchedContainer function, send out direction, distance and type of cones (camera cone) via od4.send


### backwardDetection: Lidar + ORB + CNN
- take image frame and lidar frame as input
- match image frame and lidar frame by timestamp
- create an OpenCV ORB detector, set nfeatures = 200, also set some threshold and patchsize
- detect keypoints
- triangulate to 3D world
- filter out points that are outside interest region
- group 3D points and compute median for each group
- project 3D points back to image
- match lidar cone (obtained by using Lidar and processing pointcloud, cone data would be more precise) and camera cone (obtained by ORB detector, cone data would be less precise than lidar one) by computing the 3D distance between lidar-cone position and camera-cone position, i.e. for each lidar cone, find the closest camera cone
- re-calibrate the calibration between lidar and camera
- take position from lidar cone, and extract RoI (centered at the median point) from the matched camera cone
- classify the RoI by CNN, find color label based on probability
- call SendMatchedContainer, send out direction, distance and type of cones (lidar-camera matched cone) via od4.send
- if offline is true, save the processed images as n_minValue.png (n = 0,1,2,..., minValue is min timestamp difference)


### labels of CNN classification
| label name | label | comments |
| :-----: | :-----: | :-----: |
| "background" | 666 | none |
| "blue" | 1 | |
| "yellow" | 2 | |
| "orange" | 3 | small orange cone |
| "orange" | 4 | big orange cone |

According to FSG rules:
- the left borders of the track are marked with small blue cones
- the right borders of the track are marked with small yellow cones
- exit and entry lanes are marked with small orange cones
- big orange cones will be placed before and after start, finish and timekeeping lines

### forwardDetection versus backardDetection
- if lidar is working, forwardDetection will not be utilized; but the collector will keep collecting lidar cones sent by microservice sensation-attention, and backwardDetection will be called
- if lidar is not working, use forwardDetectioin
- however, only using ORB detection (forwardDetection) will have problems with delay and uneven ground and cause RoI off the cone center
- lidar provides solution to these:
  - fix delay: match between lidar and camera frame by timestamp, and between match lidar and camera cone (or RoI) by 3D distance
  - fix uneven ground: real time calibration between lidar and camera
  - provide accurate position: position from lidar rather than ORB, classification from camera


### Questions
- the image used in ORB detection, is from left camera or right camera or a merged one (since we are using stereo camera)?
- what does "delay" mean when not using lidar?
- why not using lidar directly (I guess we are able to use lidar RoI to make CNN classification)?
- ...


### To-do
- Refactor the name of command arguments, e.g. the "sender stamp" in perception-detectcone is named senderStamp, in sensation-attention is named "id"
- Refactor function names, as some functions start with uppercase letter while others start with lowercase letter
- ...