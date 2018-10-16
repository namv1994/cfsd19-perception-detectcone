# cfsd18-perception-detectcone

[![Build Status](https://travis-ci.org/cfsd/cfsd18-perception-detectcone.svg?branch=master)](https://travis-ci.org/cfsd/cfsd18-perception-detectcone)

### Command line arguments
| parameter | comments |
| ----- | ----- |
| cid | OpenDaVINCI v4 session identifier [1..254] |
| name | name of the shared memory area; must start with '/' and be no longer than 255, if not it will be adjusted accordingly |
| width | image width, image is from camera |
| height | image height |
| bpp | bits per pixel, should be either 24 or 8 |
| threshold | |
| timeDiffMilliseconds | |
| separationTimeMs | |
| checkLidarMilliseconds | |
| senderStamp | sender stamp of this microservice |
| attentionSenderStamp | the sender stamp of sensation-attention, is used to receive the desired envelope |
| offline | if true, will not save data from shared memory as png images, but will use the already existed png images for further processing |
| annotate | |
| stateMachineId | |
| readyStateMachine | |
| forwardDetection | if ture, will call forwardDetectionORB |
| fastThreshold | |
| matchDistance | |


### OD4Session message in and out
- receive:
  - opendlv.logic.perception.ObjectDirection (receiver stamp: 116, from sensation-attention)
  - opendlv.logic.perception.ObjectDistance (receiver stamp: 116, from sensation-attention)
  - opendlv.proxy.SwitchStateReading (from ?)
- send:
  - opendlv.system.SignalStatusMessage (sender stamp: 118)
  - opendlv.logic.perception.ObjectDirection (sender stamp: 118)
  - opendlv.logic.perception.ObjectDistance (sender stamp: 118)
  - opendlv.logic.perception.ObjectType (sender stamp: 118)
  - opendlv.logic.perception.Object (sender stamp: 118)


### Function call graph


### main function


### forwardDetectionORB
- ORB + CNN
- create an OpenCV ORB detector, and set the maximum number of featrues to retain as 100
- ORB detects keypoints within a specific row range of the image (does the image from left camera or right camera or a merged one?)
- 2D position from keypoints, 3D points projected from 2D positions
- discard the 3D keypoints that are outside the effective region, and then filter the rest 3D keypoints further
- map the filtered 3D points to 2D points, and extract RoI (which is cv::Rect)
- patch the image with the RoI
- convert the patched image to tiny_dnn inputs
- given the inputs, predict the probability of each color with pre-trained tiny-dnn model
- find the maxProb and the corresponding position for each cone
- if annotate is true, make annotation
- save the processed image with results as png file via an independent thread
- call SendMatchedContainer function, send out cone direction, distance, type via OD4Session


### backwardDetection
- Lidar + ORB + CNN
- Collector collects information of cones send by sensation-attention in which Lidar processing algorithms are implemented.
- the collected cones are passed to backwardDetection
-

### Questions
- stereo camera, there should be two images, are they merged and then written to shared memory?
-

### To-do
- Refactor the name of command arguments, e.g. the "sender stamp" in perception-detectcone is named senderStamp, in sensation-attention is named "id"
- 