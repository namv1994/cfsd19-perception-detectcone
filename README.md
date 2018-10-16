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


### forwardDetectionORB
- ORB + CNN
- OpenCV: ORB, keypoints detection
- 2D position from keypoints, 3D points projected from 2D positions
- effective region for 3D points, and also have to filter the rest key 3D points
- map the filtered 3D points to 2D points, and extract RoI
- patch the image with RoI
- convert the patched image to tiny_dnn inputs
- predict the probability of the inputs with pre-trained tiny-dnn model
- find the maxProb and the corresponding position
- filter cones

### backwardDetection
- Lidar + ORB + CNN
- Collector collects information of cones send by sensation-attention in which Lidar processing algorithms are implemented.
- the collected cones are passed to backwardDetection
-

### Questions
- stereo camera, but it seems only one image is to be processed?
-

### To-do
- Refactor the name of command arguments, e.g. the "sender stamp" in perception-detectcone is named senderStamp, in sensation-attention is named "id"
-