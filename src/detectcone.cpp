/**
* Copyright (C) 2017 Chalmers Revere
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
* USA.
*/
#include "detectcone.hpp"

DetectCone::DetectCone(std::map<std::string, std::string> commandlineArguments, cluon::OD4Session& od4) :
  m_od4(od4)
, m_threshold(0.9f)
, m_coneTimeStamp()
, m_imgTimeStamp()
, m_coneCollector()
, m_lastObjectId(0)
, m_coneMutex()
, m_imgMutex()
, m_stateMachineMutex()
, m_recievedFirstImg(false)
, m_img()
, m_imgAndTimeStamps()
, m_timeStamps()
, m_count(0)
, m_count2(0)
, m_currentFrame(0)
, m_offline(false)
, m_annotate(false)
, m_verbose(false)
, m_forwardDetection(false)
, m_drivingState(false)
, m_readyStateMachine()
, m_model()
, m_lidarIsWorking(false)
, m_checkLidarMilliseconds()
, m_patchSize(64)
, m_width(672)
, m_height(376)
, m_xShift(0)//0
, m_yShift(0.9)//0.833
, m_zShift(1.1)//1.872
, m_fastThreshold(20)
, m_matchDistance(1.5)
, m_orbPatchSize(31)
, m_folderName()
, m_maxZ(8.2f)//8.1234
, m_file()
{
  m_coneCollector = Eigen::MatrixXd::Zero(4,2000);
  setUp(commandlineArguments);
}

DetectCone::~DetectCone()
{
  m_file.close();
}

void DetectCone::setUp(std::map<std::string, std::string> commandlineArguments)
{
  m_threshold = static_cast<float>(std::stof(commandlineArguments["threshold"]));
  m_checkLidarMilliseconds = static_cast<int64_t>(std::stoi(commandlineArguments["checkLidarMilliseconds"]));
  m_senderStamp = static_cast<uint32_t>(std::stoi(commandlineArguments["senderStamp"]));
  m_annotate = static_cast<bool>(std::stoi(commandlineArguments["annotate"]));
  m_readyStateMachine = static_cast<bool>(std::stoi(commandlineArguments["readyStateMachine"]));
  m_verbose = static_cast<bool>(commandlineArguments.count("verbose") != 0);
  m_forwardDetection = static_cast<bool>(std::stoi(commandlineArguments["forwardDetection"]));
  m_fastThreshold = static_cast<uint32_t>(std::stoi(commandlineArguments["fastThreshold"]));
  m_matchDistance = static_cast<float>(std::stof(commandlineArguments["matchDistance"]));

  if(m_annotate){
    std::string fileName = "/opt/annotations/", command;
    std::ifstream infile(fileName);
    if(infile.good()){
        command = "rm -r "+fileName;
        system(command.c_str());
    }
    command = "mkdir /opt/annotations";
    system(command.c_str());
    command = "mkdir /opt/annotations/0";
    system(command.c_str());
    command = "mkdir /opt/annotations/1";
    system(command.c_str());
    command = "mkdir /opt/annotations/2";
    system(command.c_str());
    command = "mkdir /opt/annotations/3";
    system(command.c_str());
  }

  CNN("model", m_model);
}

void DetectCone::getFolderName(const std::string folderName){
  m_folderName = folderName+"/";

  int index = folderName.find_last_of('/');
  std::string filepathTimestamp = folderName.substr(0, index) + "/log.txt";
  m_file.open(filepathTimestamp.c_str());
}

void DetectCone::receiveCombinedMessage(cluon::data::TimeStamp currentFrameTime,std::map<int,ConePackage> currentFrame){
  m_coneTimeStamp = currentFrameTime;
  Eigen::MatrixXd cones = Eigen::MatrixXd::Zero(4,currentFrame.size());
  std::map<int,ConePackage>::iterator it;
  m_start = cluon::time::now();
  int coneIndex = 0;
  it =currentFrame.begin();
  while(it != currentFrame.end()){
    auto direction = std::get<0>(it->second);
    auto distance = std::get<1>(it->second);
    if(std::isnan(direction.azimuthAngle()) || std::isnan(direction.zenithAngle()) || std::isnan(distance.distance())){
      std::cout << "Nan xyz: " << m_currentFrame << std::endl;
    }
    else{
      cones(0,coneIndex) = -direction.azimuthAngle();
      cones(1,coneIndex) = direction.zenithAngle();
      cones(2,coneIndex) = distance.distance();
      coneIndex++;
    }
    it++;
  }
  if(cones.cols()>0 && m_recievedFirstImg){
    if(m_verbose)
      std::cout << "received " << cones.cols() << " cones" << std::endl;
    SendCollectedCones(cones);
  }
}

void DetectCone::nextContainer(cluon::data::Envelope data)
{
  if (data.dataType() == opendlv::logic::perception::ObjectDirection::ID()) {
    m_coneTimeStamp = data.sampleTimeStamp();
    auto coneDirection = cluon::extractMessage<opendlv::logic::perception::ObjectDirection>(std::move(data));
    uint32_t objectId = coneDirection.objectId();
    std::unique_lock<std::mutex> lockCone(m_coneMutex);
    m_coneCollector(0,objectId) = -coneDirection.azimuthAngle();  //Negative for conversion from car to LIDAR frame
    m_coneCollector(1,objectId) = coneDirection.zenithAngle();
    m_lastObjectId = (m_lastObjectId<objectId)?(objectId):(m_lastObjectId);
  }

  else if(data.dataType() == opendlv::logic::perception::ObjectDistance::ID()){
    m_coneTimeStamp = data.sampleTimeStamp();
    auto coneDistance = cluon::extractMessage<opendlv::logic::perception::ObjectDistance>(std::move(data));
    uint32_t objectId = coneDistance.objectId();
    std::unique_lock<std::mutex> lockCone(m_coneMutex);
    m_coneCollector(2,objectId) = coneDistance.distance();
    m_coneCollector(3,objectId) = 0;
    m_lastObjectId = (m_lastObjectId<objectId)?(objectId):(m_lastObjectId);
  }
}

void DetectCone::setStateMachineStatus(cluon::data::Envelope data){
  std::lock_guard<std::mutex> lockStateMachine(m_stateMachineMutex);
  auto machineStatus = cluon::extractMessage<opendlv::proxy::SwitchStateReading>(std::move(data));
  int state = machineStatus.state();
  if(state == 2){
    m_drivingState = true;
  }
  else{
    m_drivingState = false;
  }
}

bool DetectCone::getdrivingState(){
  if(m_readyStateMachine){
    m_drivingState = true;
  }
  return m_drivingState;
}

void DetectCone::setTimeStamp(std::pair<int64_t, cv::Mat> imgAndTimeStamp){
  if(m_imgAndTimeStamps.size()>20){
    m_imgAndTimeStamps.clear();
  }
  m_imgAndTimeStamps.push_back(imgAndTimeStamp);

  m_imgTimeStamp = imgAndTimeStamp.first;
  m_img = imgAndTimeStamp.second;

  m_recievedFirstImg = true;
  m_count++;
}

void DetectCone::getTimeStamp(const std::string path){
  std::string line;
  m_timeStamps.clear();
  std::ifstream f(path);
  if (f.is_open()){
    while (getline(f,line)){
      int64_t timeStamp = static_cast<int64_t>(std::stol(line));
      m_timeStamps.push_back(timeStamp);
    }
    f.close();
  }
  else{
    if(m_verbose)
      std::cout << "Unable to open timestamp file" << std::endl;
  }

  m_recievedFirstImg = true;
  m_offline = true;
}

void DetectCone::checkLidarState(){
  if(m_offline){
    if(cluon::time::toMicroseconds(m_coneTimeStamp) > 0){
      m_lidarIsWorking = true;
    }
    else{
      if(m_verbose)
        std::cout << "currentFrame: " << m_currentFrame << std::endl;
      m_lidarIsWorking = false;
      if(m_forwardDetection){
        m_img = cv::imread("/opt/images/"+std::to_string(m_currentFrame++)+".png");
        forwardDetectionORB(m_img);
      }
    }
  }
  else{
    int64_t timeDiff = m_imgTimeStamp - cluon::time::toMicroseconds(m_coneTimeStamp);
    if ((timeDiff > m_checkLidarMilliseconds*1000)){
      if(m_verbose)
        std::cout << "No lidar data received" << std::endl;
      m_lidarIsWorking = false;
      if(m_img.empty()){
        return;
      }
      forwardDetectionORB(m_img);
    }
    else{
      m_lidarIsWorking = true;
    }
  }
}

void DetectCone::blockMatching(cv::Mat& disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR, dispL, dispR;

  cv::cvtColor(imgL, grayL, 6);
  cv::cvtColor(imgR, grayR, 6);

  cv::Ptr<cv::StereoBM> sbmL = cv::StereoBM::create();
  sbmL->setBlockSize(19);
  sbmL->setNumDisparities(32);
  sbmL->setUniquenessRatio(15);
  sbmL->compute(grayL, grayR, dispL);

  disp = dispL/16;
}

void DetectCone::reconstruction(cv::Mat img, cv::Mat& Q, cv::Mat& rectified, cv::Mat& XYZ){
  cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
    349.891, 0, 334.352,
    0, 349.891, 187.937,
    0, 0, 1);
  cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.173042, 0.0258831, 0, 0, 0);
  cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
    350.112, 0, 345.88,
    0, 350.112, 189.891,
    0, 0, 1);
  cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.174209, 0.026726, 0, 0, 0);
  cv::Mat rodrigues = (cv::Mat_<double>(3, 1) << -0.0132397, 0.021005, -0.00121284);

  cv::Mat R;
  cv::Rodrigues(rodrigues, R);
  cv::Mat T = (cv::Mat_<double>(3, 1) << -0.12, 0, 0);
  cv::Size stdSize = cv::Size(m_width, m_height);

  int width = img.cols;
  int height = img.rows;
  cv::Mat imgL(img, cv::Rect(0, 0, width/2, height));
  cv::Mat imgR(img, cv::Rect(width/2, 0, width/2, height));

  cv::resize(imgL, imgL, stdSize);
  cv::resize(imgR, imgR, stdSize);

  cv::Mat R1, R2, P1, P2;
  cv::Rect validRoI[2];
  cv::stereoRectify(mtxLeft, distLeft, mtxRight, distRight, stdSize, R, T, R1, R2, P1, P2, Q,
    cv::CALIB_ZERO_DISPARITY, 0.0, stdSize,& validRoI[0],& validRoI[1]);

  cv::Mat rmap[2][2];
  cv::initUndistortRectifyMap(mtxLeft, distLeft, R1, P1, stdSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  cv::initUndistortRectifyMap(mtxRight, distRight, R2, P2, stdSize, CV_16SC2, rmap[1][0], rmap[1][1]);
  cv::remap(imgL, imgL, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
  cv::remap(imgR, imgR, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);

  cv::Mat disp;
  blockMatching(disp, imgL, imgR);

  imgL.copyTo(rectified);

  cv::reprojectImageTo3D(disp, XYZ, Q);
}

void DetectCone::convertImage(cv::Mat img, int w, int h, tiny_dnn::vec_t& data){
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(w, h));
  data.resize(w * h * 3);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
       data[c * w * h + y * w + x] =
         float(resized.at<cv::Vec3b>(y, x)[c] / 255.0);
      }
    }
  }
}

void DetectCone::CNN(const std::string& dictionary, tiny_dnn::network<tiny_dnn::sequential>& model) {
  using conv    = tiny_dnn::convolutional_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using tanh    = tiny_dnn::tanh_layer;
  using relu    = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  model << conv(64, 64, 4, 3, 16, tiny_dnn::padding::valid, true, 2, 2, 1, 1, backend_type) << tanh()
     << conv(31, 31, 3, 16, 16, tiny_dnn::padding::valid, true, 2, 2,  1, 1, backend_type) << tanh()
     << conv(15, 15, 3, 16, 32, tiny_dnn::padding::valid, true, 2, 2, 1, 1, backend_type) << tanh()
     << conv(7, 7, 3, 32, 32, tiny_dnn::padding::valid, true, 2, 2, 1, 1, backend_type) << tanh()
     << fc(3 * 3 * 32, 128, true, backend_type) << relu()
     << fc(128, 4, true, backend_type) << softmax(4);

  std::ifstream ifs(dictionary.c_str());
  if (!ifs.good()){
    if(m_verbose)
      std::cout << "CNN model does not exist!" << std::endl;
    return;
  }
  ifs >> model;
}

void DetectCone::imRegionalMax(std::vector<Cone>& cones, size_t label, cv::Mat input, int nLocMax, double threshold, int minDistBtwLocMax)
{
    cv::Mat scratch = input.clone();
    for (int i = 0; i < nLocMax; i++) {
        cv::Point location;
        double maxVal;
        cv::minMaxLoc(scratch, NULL, &maxVal, NULL, &location);
        if (maxVal > threshold) {
            int col = location.x;
            int row = location.y;
            Cone cone = Cone(0.0,0.0,0.0);
            cone.m_pt = cv::Point(col, row);
            cone.m_prob = maxVal;
            cone.m_label = label;
            cones.push_back(cone);
            int r0 = (row-minDistBtwLocMax > -1 ? row-minDistBtwLocMax : 0);
            int r1 = (row+minDistBtwLocMax < scratch.rows ? row+minDistBtwLocMax : scratch.rows-1);
            int c0 = (col-minDistBtwLocMax > -1 ? col-minDistBtwLocMax : 0);
            int c1 = (col+minDistBtwLocMax < scratch.cols ? col+minDistBtwLocMax : scratch.cols-1);
            for (int r = r0; r <= r1; r++) {
                for (int c = c0; c <= c1; c++) {
                    if (sqrt((r-row)*(r-row)+(c-col)*(c-col)) <= minDistBtwLocMax) {
                      scratch.at<double>(r,c) = 0.0;
                    }
                }
            }
        } else {
            break;
        }
    }
}


cv::Point3f DetectCone::median(std::vector<cv::Point3f> vec3) {
  size_t size = vec3.size();
  float tvecan[3];
  std::vector<float> vec[3];
  for (size_t i = 0; i < size; i++){
    vec[0].push_back(vec3[i].x);
    vec[1].push_back(vec3[i].y);
    vec[2].push_back(vec3[i].z);
  }

  for (size_t i = 0; i < 3; i++){
    std::sort(vec[i].begin(), vec[i].end());
    if (size % 2 == 0) {
      tvecan[i] = (vec[i][size/2-1] + vec[i][size/2])/2;
    }
    else
      tvecan[i] = vec[i][size/2];
  }

  return cv::Point3f(tvecan[0],0.9f,tvecan[2]);
}

void DetectCone::gather_points(
  cv::Mat source,
  std::vector<float> vecQuery,
  std::vector<int>& vecIndex,
  std::vector<float>& vecDist
  )
{
  double radius = 0.5;
  unsigned int max_neighbours = 100;
  cv::flann::KDTreeIndexParams indexParams(2);
  cv::flann::Index kdtree(source, indexParams);
  cv::flann::SearchParams params(1024);
  kdtree.radiusSearch(vecQuery, vecIndex, vecDist, radius, max_neighbours, params);
}

void DetectCone::filterKeypoints(std::vector<cv::Point3f>& point3Ds){
  std::vector<Pt> data;

  for(size_t i = 0; i < point3Ds.size(); i++){
    data.push_back(Pt{point3Ds[i],-1});
  }

  cv::Mat source = cv::Mat(point3Ds).reshape(1);
  point3Ds.clear();
  cv::Point3f point3D;
  int groupId = 0;

  for(size_t j = 0; j < data.size()-1; j++)
  {
    if(data[j].group == -1){
      std::vector<float> vecQuery(3);
      vecQuery[0] = data[j].pt.x;
      vecQuery[1] = data[j].pt.y;
      vecQuery[2] = data[j].pt.z;
      std::vector<int> vecIndex;
      std::vector<float> vecDist;

      gather_points(source, vecQuery, vecIndex, vecDist);
      int num = 0;
      for(size_t i = 0; i < vecIndex.size(); i++){
        if(vecIndex[i]!=0)
          num++;
      }
      for (size_t i = 1; i < vecIndex.size(); i++){
        if (vecIndex[i] == 0 && vecIndex[i+1] != 0){
          num++;
        }
      }
      if (num == 0){
        if (data[j].group == -1){
          data[j].group = groupId++;
          point3D = data[j].pt;
        }
      }
      else{
        std::vector<Pt> groupAll;
        std::vector<int> filteredIndex;
        std::vector<cv::Point3f> centerPoints;
        for (int v = 0; v < num; v++){
          groupAll.push_back(data[vecIndex[v]]);
          filteredIndex.push_back(vecIndex[v]);
        }

        int noGroup = 0;
        for(size_t i = 0; i < groupAll.size(); i++){
          if(groupAll[i].group == -1)
            noGroup++;
        }

        if (noGroup > 0){
          for (size_t k = 0; k < filteredIndex.size(); k++)
          {
            if (data[filteredIndex[k]].group == -1)
            {
              data[filteredIndex[k]].group = groupId;
              centerPoints.push_back(data[vecIndex[k]].pt);
            }
          }
          groupId++;
          point3D = median(centerPoints);
        }
        else{
          data[j].group = data[vecIndex[0]].group;
          point3D = data[j].pt;
        }
      }
      if(std::isnan(point3D.x) || std::isnan(point3D.y) || std::isnan(point3D.z) || point3D.z >= m_maxZ)
        continue;
      point3Ds.push_back(point3D);
    }
  }
}

int DetectCone::xyz2xy(cv::Mat Q, cv::Point3f xyz, cv::Point& xy, float radius){
  float X = xyz.x;
  float Y = xyz.y;
  float Z = xyz.z;
  float Cx = float(-Q.at<double>(0,3));
  float Cy = float(-Q.at<double>(1,3));
  float f = float(Q.at<double>(2,3));
  float a = float(Q.at<double>(3,2));
  float b = float(Q.at<double>(3,3));
  float d = (f - Z * b ) / ( Z * a);
  xy.x = int(X * ( d * a + b ) + Cx);
  xy.y = int(Y * ( d * a + b ) + Cy);
  return int(radius * ( d * a + b ));
}

int DetectCone::countFiles(const char* path){
  struct dirent *de;
  DIR *dir = opendir(path);
  if(!dir)
  {
    return -1;
  }

  unsigned long count = 0;
  while((de = readdir(dir)))
  {
    ++count;
  }

  closedir(dir);
  return count;
}

void DetectCone::annotate(cv::Mat img, int maxIndex, cv::Point position, int radius){
  std::string path = "/opt/annotations/"+std::to_string(maxIndex);
  int num = countFiles(path.c_str());
  path += "/"+std::to_string(maxIndex)+"_"+std::to_string(num)+".png";

  cv::Rect roi;
  roi.x = std::max(position.x - radius, 0);
  roi.y = std::max(position.y - radius, 0);
  roi.width = std::min(position.x + radius, img.cols) - roi.x;
  roi.height = std::min(position.y + radius, img.rows) - roi.y;
  if(0 < roi.width && 0 < roi.height && radius > 0){
    cv::Mat patchImg = img(roi);
    cv::resize(patchImg, patchImg, cv::Size(m_patchSize,m_patchSize));
    cv::imwrite(path, patchImg);
  }
}

void DetectCone::forwardDetectionORB(cv::Mat img){
  //Given RoI by ORB detector and detected by CNN
  cluon::data::TimeStamp timestamp = cluon::time::now();
  // std::lock_guard<std::mutex> lockStateMachine(m_stateMachineMutex);
  if(!m_drivingState)
  {
    return;
  }
  std::vector<Cone> cones;
  std::vector<tiny_dnn::tensor_t> inputs;
  std::vector<int> verifiedIndex;
  std::vector<cv::Point> candidates;
  std::vector<cv::Scalar> colors;
  colors.push_back(cv::Scalar(0,0,0));
  colors.push_back(cv::Scalar(255,0,0));
  colors.push_back(cv::Scalar(0,255,255));
  colors.push_back(cv::Scalar(0,165,255));
  std::string labels[] = {"background", "blue", "yellow", "orange"};
  int resultWidth = m_height;
  int resultHeight = m_height;
  cv::Mat result = cv::Mat::zeros(resultWidth,resultHeight,CV_8UC3);
  double resultResize = 15;

  cv::Mat Q, XYZ, imgRoI, imgSource;
  reconstruction(img, Q, img, XYZ);

  img.copyTo(imgSource);
  int rowT = 190;
  int rowB = 320;
  imgRoI = imgSource.rowRange(rowT, rowB);

  cv::Ptr<cv::ORB> detector = cv::ORB::create(100);
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(imgRoI, keypoints);
  if(keypoints.size()==0)
    return;

  cv::Mat probMap[4] = cv::Mat::zeros(m_height, m_width, CV_64F);

  std::vector<cv::Point3f> point3Ds;
  cv::Point point2D;
  std::vector<cv::Point> positions;
  for(size_t i = 0; i < keypoints.size(); i++){
    cv::Point position(int(keypoints[i].pt.x), int(keypoints[i].pt.y)+rowT);
    cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
    if(point3D.y>0.7 && point3D.y<0.9 && point3D.z > 0 && point3D.z < m_maxZ){
      point3Ds.push_back(point3D);
      positions.push_back(position);
    }
  }
  if(point3Ds.size()==0)
    return;
  filterKeypoints(point3Ds);
  for(size_t i = 0; i < point3Ds.size(); i++){
    int radius = xyz2xy(Q, point3Ds[i], point2D, 0.3f);
    int x = point2D.x;
    int y = point2D.y;

    cv::Rect roi;
    roi.x = std::max(x - radius, 0);
    roi.y = std::max(y - radius, 0);
    roi.width = std::min(x + radius, img.cols) - roi.x;
    roi.height = std::min(y + radius, img.rows) - roi.y;

    if(0 < roi.width && 0 < roi.height && radius > 0){
      cv::Mat patchImg = imgSource(roi);
      tiny_dnn::vec_t data;
      convertImage(patchImg, m_patchSize, m_patchSize, data);
      inputs.push_back({data});
      verifiedIndex.push_back(i);
      candidates.push_back(cv::Point(x,y));
    }
  }

  if(inputs.size()>0){
    auto prob = m_model.predict(inputs);
    for(size_t i = 0; i < inputs.size(); i++){
      size_t maxIndex = 0;
      double maxProb = prob[i][0][0];
      for(size_t j = 1; j < 4; j++){
        if(prob[i][0][j] > maxProb){
          maxIndex = j;
          maxProb = prob[i][0][j];
        }
      }
      int x = candidates[i].x;
      int y = candidates[i].y;
      if(maxIndex > 0)
        probMap[maxIndex].at<double>(y,x) = maxProb;
    }
    for(size_t i = 0; i < 4; i++){
      imRegionalMax(cones, i, probMap[i], 10, m_threshold, 10);
    }

    for(size_t i = 0; i < cones.size(); i++){
      int x = cones[i].m_pt.x;
      int y = cones[i].m_pt.y;
      double maxProb = cones[i].m_prob;
      int maxIndex = cones[i].m_label;
      cv::Point position(x, y);
      cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
      std::string labelName = labels[maxIndex];
      cv::Point position_tmp;
      int radius = xyz2xy(Q, point3D, position_tmp, 0.3f);

      if(radius<=0){
        cones[i].m_label = 0;
        continue;
      }

      if(m_annotate){
        annotate(imgSource, maxIndex, position, radius);
      }

      cones[i].setX(point3D.x-m_xShift);
      cones[i].setY(point3D.z-m_zShift);
      cones[i].setZ(point3D.y-m_yShift);

      int xt = int(point3D.x * float(resultResize) + resultWidth/2);
      int yt = int(point3D.z * float(resultResize));
      if(m_verbose){
        std::cout << "label: " << labels[maxIndex] << ", probability: " << maxProb << std::endl;
      }
      cv::circle(img, position, radius, colors[maxIndex], 2);
      if (xt >= 0 && xt <= resultWidth && yt >= 0 && yt <= resultHeight){
        cv::circle(result, cv::Point (xt,yt), 6, colors[maxIndex], -1);
      }
    }
  }

  for(size_t i = 0; i < positions.size(); i++){
    cv::circle(img, positions[i], 2, cv::Scalar (255,255,255), -1);
  }

  cv::line(img, cv::Point(0,rowT), cv::Point(m_width,rowT), cv::Scalar(0,0,255), 2);
  cv::line(img, cv::Point(0,rowB), cv::Point(m_width,rowB), cv::Scalar(0,0,255), 2);

  cv::Mat outImg;
  cv::flip(result, result, 0);
  cv::hconcat(img,result,outImg);

  std::string saveString = m_folderName+std::to_string(m_currentFrame)+".png";
  std::thread imWriteThread(&DetectCone::saveImages,this,saveString,outImg);
  imWriteThread.detach();
  double timeDiff = (cluon::time::toMicroseconds(cluon::time::now()) - cluon::time::toMicroseconds(timestamp))/1000;
  // if(m_verbose)
  std::cout << "forward detection time: " << timeDiff << "ms" << std::endl;
  std::vector<Cone> conesToSend = MatchCones(cones);
  SendMatchedContainer(conesToSend);
}

void DetectCone::backwardDetection(cv::Mat img, Eigen::MatrixXd& lidarCones, int64_t minValue){
  //Given RoI in 3D world, project back to the camera frame and then detect
  cluon::data::TimeStamp timestamp = cluon::time::now();
  cv::Mat Q, XYZ, imgSource;
  reconstruction(img, Q, img, XYZ);
  std::vector<tiny_dnn::tensor_t> inputs;
  std::vector<int> verifiedIndex;
  std::vector<cv::Vec3i> porperty;
  std::vector<Cone> localCones;
  std::vector<cv::Scalar> colors;
  colors.push_back(cv::Scalar(0,0,0));
  colors.push_back(cv::Scalar(255,0,0));
  colors.push_back(cv::Scalar(0,255,255));
  colors.push_back(cv::Scalar(0,165,255));
  std::string labels[] = {"background", "blue", "yellow", "orange"};
  std::vector<double> xDiffs, yDiffs, zDiffs;

  int resultWidth = m_height;
  int resultHeight = m_height;
  cv::Mat result = cv::Mat::zeros(resultWidth,resultHeight,CV_8UC3);
  double resultResize = 15;

  // cv::medianBlur(img, img, 5);
  img.copyTo(imgSource);
  int rowT = 190;
  int rowB = 320;
  cv::Mat imgRoI = imgSource.rowRange(rowT, rowB);

  cv::Ptr<cv::ORB> detector = cv::ORB::create(200);
  detector->setFastThreshold(m_fastThreshold);
  detector->setPatchSize(m_orbPatchSize);
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(imgRoI, keypoints);
  if(keypoints.size() == 0){
    return;
  }

  std::vector<cv::Point3f> point3Ds;
  std::vector<cv::Point> positions;
  for(size_t i = 0; i < keypoints.size(); i++){
    cv::Point position(int(keypoints[i].pt.x), int(keypoints[i].pt.y)+rowT);
    cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
    if(point3D.y > 0.7 && point3D.y < 0.9 && point3D.z > 0 && point3D.z < m_maxZ){
      point3Ds.push_back(point3D);
      positions.push_back(position);
    }
  }
  if(point3Ds.size()==0)
    return;
  filterKeypoints(point3Ds);

  for(int i = 0; i < lidarCones.cols(); i++){
    cv::Point point2D, cameraPoint2D;
    Cone cone = Cone(lidarCones(0,i),lidarCones(1,i),lidarCones(2,i));
    cone.m_label = 0;
    cv::Point3f lidarCone(float(m_xShift+lidarCones(0,i)), float(m_yShift+lidarCones(2,i)), float(m_zShift+lidarCones(1,i)));
    int radius = xyz2xy(Q, lidarCone, point2D, 0.6f);
    int x = point2D.x;
    int y = point2D.y;
    cv::Rect roi;
    roi.x = std::max(x - radius, 0);
    roi.y = std::max(y - radius, 0);
    roi.width = std::min(std::max(x + radius,0), img.cols) - roi.x;
    roi.height = std::min(std::max(y + radius,0), img.rows) - roi.y;

    // radius = xyz2xy(Q, lidarCone, point2D, 0.3f);
    if(0 < roi.width && 0 < roi.height && radius > 0){
      if(m_offline)
        cv::circle(img, cv::Point(x,y), radius, cv::Scalar (255,255,255), 2);
      int minIndex = -1;
      float minDistance = 1.5f;
      for(size_t j = 0; j < point3Ds.size(); j++){
        int radius_tmp = xyz2xy(Q, point3Ds[j], cameraPoint2D, 0.3f);
        if(cameraPoint2D.x >= roi.x && cameraPoint2D.x <= roi.x+roi.width && cameraPoint2D.y >= roi.y && cameraPoint2D.y <= roi.y+roi.height){
          float distance = static_cast<float>(std::pow(std::pow(point3Ds[j].x-lidarCone.x,2)+std::pow(point3Ds[j].y-lidarCone.y,2)+std::pow(point3Ds[j].z-std::min(m_maxZ, lidarCone.z),2),0.5));
          if(distance<minDistance){
            minIndex = int(j);
            minDistance = distance;
            radius = radius_tmp;
            point2D = cameraPoint2D;
          }
        }
      }
      if(minIndex > -1){
        // m_file << lidarCones(0,i) << " " << lidarCones(2,i) << " " << lidarCones(1,i) << " " << point3Ds[minIndex].x << " " << point3Ds[minIndex].y << " " << point3Ds[minIndex].z << std::endl;
        if(point3Ds[minIndex].z < 8.1234f){
          xDiffs.push_back(point3Ds[minIndex].x-lidarCones(0,i));
          yDiffs.push_back(point3Ds[minIndex].y-lidarCones(2,i));
          zDiffs.push_back(point3Ds[minIndex].z-lidarCones(1,i));
        }
        point3Ds.erase(point3Ds.begin() + minIndex);
        x = int(point2D.x);
        y = int(point2D.y);
        roi.x = std::max(x - radius, 0);
        roi.y = std::max(y - radius, 0);
        roi.width = std::min(std::max(x + radius,0), img.cols) - roi.x;
        roi.height = std::min(std::max(y + radius,0), img.rows) - roi.y;

        if(0 < roi.width && 0 < roi.height && radius > 0){
          cv::Mat patchImg = imgSource(roi);
          tiny_dnn::vec_t data;
          convertImage(patchImg, m_patchSize, m_patchSize, data);
          inputs.push_back({data});
          verifiedIndex.push_back(i);
          porperty.push_back(cv::Vec3i(x,y,radius));
        }
      }
    }
    localCones.push_back(cone);
  }

  if(xDiffs.size()>0){
    double sum = std::accumulate(std::begin(xDiffs), std::end(xDiffs), 0.0);
    if(std::abs(0-sum/xDiffs.size())<0.3)
      m_xShift = (m_xShift+sum/xDiffs.size())/2;
    sum = std::accumulate(std::begin(yDiffs), std::end(yDiffs), 0.0);
    if(std::abs(0.9-sum/yDiffs.size())<0.3)
      m_yShift = (m_yShift+sum/yDiffs.size())/2;
    sum = std::accumulate(std::begin(zDiffs), std::end(zDiffs), 0.0);
    if(std::abs(1.1-sum/zDiffs.size())<0.5)
      m_zShift = (m_zShift+sum/zDiffs.size())/2;
    m_file << m_xShift << " " << m_yShift << " " << m_zShift << std::endl;
  }
  else{
    m_xShift = 0;
    m_yShift = 0.9;
    m_zShift = 1.1;
  }

  if(inputs.size()>0){
    auto prob = m_model.predict(inputs);
    for(size_t i = 0; i < inputs.size(); i++){
      size_t maxIndex = 0;
      float_t maxProb = prob[i][0][0];
      for(size_t j = 1; j < 4; j++){
        if(prob[i][0][j] > maxProb){
          maxIndex = j;
          maxProb = prob[i][0][j];
        }
      }

      cv::Point position(porperty[i][0],porperty[i][1]);
      int radius = porperty[i][2];

      if(m_annotate){
        annotate(imgSource, maxIndex, position, radius);
      }

      localCones[verifiedIndex[i]].m_prob = maxProb;
      if (maxIndex > 0 && maxProb > m_threshold){
        localCones[verifiedIndex[i]].m_label = maxIndex;
      }
      if(maxIndex == 3)
        localCones[verifiedIndex[i]].m_label = 4;
      std::string labelName = labels[maxIndex];
      if(m_verbose){
        std::cout << "label: " << labels[maxIndex] << ", probability: " << maxProb << std::endl;
      }
      if(m_offline)
        cv::circle(img, position, radius, colors[maxIndex], 2);
    }
  }
  std::vector<Cone> conesToSend = MatchCones(localCones);
  SendMatchedContainer(conesToSend);

  // SendMatchedContainer(localCones);
  if(m_offline){
    for(int i = 0; i < lidarCones.cols(); i++){
      cv::Point3f lidarCone(float(m_xShift+lidarCones(0,i)), float(m_yShift+lidarCones(2,i)), float(m_zShift+lidarCones(1,i)));
      int xt = int(lidarCone.x * float(resultResize) + resultWidth/2);
      int yt = int(lidarCone.z * float(resultResize));
      if (xt >= 0 && xt <= resultWidth && yt >= 0 && yt <= resultHeight){
        cv::circle(result, cv::Point (xt,yt), 6, cv::Scalar (255,255,255), -1);
      }
    }

    // for(size_t i = 0; i < localCones.size(); i++){
    //   int label = localCones[i].getLabel();
    //   if(label == 4)
    //     label = 3;
    //   int xt = int((localCones[i].getX() + m_xShift) * float(resultResize) + resultWidth/2);
    //   int yt = int((localCones[i].getY() + m_zShift) * float(resultResize));
    //   if (xt >= 0 && xt <= resultWidth && yt >= 0 && yt <= resultHeight){
    //     cv::circle(result, cv::Point (xt,yt), 3, colors[label], -1);
    //   }
    // }

    for(size_t i = 0; i < conesToSend.size(); i++){
      int label = conesToSend[i].getLabel();
      if(label == 4)
        label = 3;
      int xt = int((conesToSend[i].getX() + m_xShift) * float(resultResize) + resultWidth/2);
      int yt = int((conesToSend[i].getY() + m_zShift) * float(resultResize));
      if (xt >= 0 && xt <= resultWidth && yt >= 0 && yt <= resultHeight){
        cv::circle(result, cv::Point (xt,yt), 3, colors[label], -1);
      }
    }

    for(size_t i = 0; i < positions.size(); i++){
      cv::circle(img, positions[i], 1, cv::Scalar(255,255,255), -1);
    }

    cv::Mat outImg;
    cv::flip(result, result, 0);
    cv::hconcat(img,result,outImg);

    std::string saveString = m_folderName+std::to_string(m_currentFrame++)+"_"+std::to_string(minValue)+".png";
    std::thread imWriteThread(&DetectCone::saveImages,this,saveString,outImg);
    imWriteThread.detach();
  }

  double timeDiff = (cluon::time::toMicroseconds(cluon::time::now()) - cluon::time::toMicroseconds(timestamp))/1000;
  // if(m_verbose)
  std::cout << "backward detection time: " << timeDiff << "ms" << std::endl;
  // m_file << m_currentFrame << " " << timeDiff << std::endl;
}

Eigen::MatrixXd DetectCone::Spherical2Cartesian(double azimuth, double zenimuth, double distance)
{
  double xData = distance * cos(zenimuth * static_cast<double>(DEG2RAD))*sin(azimuth * static_cast<double>(DEG2RAD));
  double yData = distance * cos(zenimuth * static_cast<double>(DEG2RAD))*cos(azimuth * static_cast<double>(DEG2RAD));
  double zData = distance * sin(zenimuth * static_cast<double>(DEG2RAD));
  Eigen::MatrixXd recievedPoint = Eigen::MatrixXd::Zero(4,1);
  recievedPoint << xData,
                   yData,
                   zData,
                    0;
  return recievedPoint;
}

void DetectCone::Cartesian2Spherical(double x, double y, double z, opendlv::logic::sensation::Point& pointInSpherical)
{
  double distance = sqrt(x*x+y*y+z*z);
  double azimuthAngle = atan2(x,y)*static_cast<double>(RAD2DEG);
  double zenithAngle = atan2(z,sqrt(x*x+y*y))*static_cast<double>(RAD2DEG);
  pointInSpherical.distance(float(distance));
  pointInSpherical.azimuthAngle(float(azimuthAngle));
  pointInSpherical.zenithAngle(float(zenithAngle));
}

void DetectCone::LidarToCoG(opendlv::logic::sensation::Point& conePoint){
  double angle = conePoint.azimuthAngle();
  double distance = conePoint.distance();
  const double lidarDistToCoG = 1.5;
  double sign = angle/std::fabs(angle);
  angle = PI - std::fabs(angle*DEG2RAD);
  double distanceNew = std::sqrt(lidarDistToCoG*lidarDistToCoG + distance*distance - 2*lidarDistToCoG*distance*std::cos(angle));
  double angleNew = std::asin((std::sin(angle)*distance)/distanceNew )*RAD2DEG;
  conePoint.azimuthAngle((float)(angleNew*sign));
  conePoint.distance((float)distanceNew);
}

void DetectCone::CameraToCoG(opendlv::logic::sensation::Point& conePoint){
  double angle = conePoint.azimuthAngle()*DEG2RAD;
  double distance = conePoint.distance();
  const double cameraDistToCoG = 0.37;
  double sign = angle/std::fabs(angle);
  double distanceNew = std::sqrt(cameraDistToCoG*cameraDistToCoG + distance*distance - 2*cameraDistToCoG*distance*std::cos(angle));
  double angleNew = std::asin((std::sin(angle)*distance)/distanceNew);
  angleNew = PI-std::fabs(angleNew);
  conePoint.azimuthAngle((float)(angleNew*sign));
  conePoint.distance((float)distanceNew);
}

void DetectCone::SendCollectedCones(Eigen::MatrixXd lidarCones)
{
  //Convert to cartesian
  std::lock_guard<std::mutex> lockStateMachine(m_stateMachineMutex);
  if(!m_drivingState)
  {
    return;
  }
  if(m_lidarIsWorking){
    Eigen::MatrixXd cone;
    for(int p = 0; p < lidarCones.cols(); p++){
      cone = Spherical2Cartesian(lidarCones(0,p), lidarCones(1,p), lidarCones(2,p));
      lidarCones.col(p) = cone;
    }
    if(m_verbose)
      std::cout << "Receive lidar data" << std::endl;
    std::unique_lock<std::mutex> lock(m_imgMutex);

    int minIndex;
    int64_t minValue;
    if(m_offline){
      minIndex = 0;
      minValue = abs(m_timeStamps[minIndex] - cluon::time::toMicroseconds(m_coneTimeStamp));
      for (size_t i = minIndex+1; i < m_timeStamps.size(); i++){
        int64_t timeDiff = abs(m_timeStamps[i]- cluon::time::toMicroseconds(m_coneTimeStamp));
        if (timeDiff<minValue){
          minIndex = i;
          minValue = timeDiff;
        }
      }
      m_img = cv::imread("/opt/images/"+std::to_string(minIndex)+".png");
      m_currentFrame = minIndex;
    }
    else{
      if (m_imgAndTimeStamps.size() == 0)
        return;
      minIndex = 0;
      minValue = abs(m_imgAndTimeStamps[minIndex].first - cluon::time::toMicroseconds(m_coneTimeStamp));
      for (size_t i = 1; i < m_imgAndTimeStamps.size(); i++){
        int64_t timeDiff = abs(m_imgAndTimeStamps[i].first- cluon::time::toMicroseconds(m_coneTimeStamp));
        if (timeDiff<minValue){
          minIndex = i;
          minValue = timeDiff;
        }
      }
      m_img = m_imgAndTimeStamps[minIndex].second;
    }
    m_currentFrame = minIndex;
    minValue /= 1000;
    if(m_verbose){
      std::cout << "minIndex: " << minIndex << ", minTimeStampDiff: " << minValue << "ms" << std::endl;
    }
    if(minValue < 100){
      backwardDetection(m_img, lidarCones, minValue);
    }
    else{
      if(m_verbose)
        std::cout << "TimeStamp not matched!" << std::endl;
    }
  }
}

std::vector<Cone> DetectCone::MatchCones(std::vector<Cone> cones){
  int m = 0;
  double posShiftX = 0;
  double posShiftY = 0;
  uint32_t numberOfCones = cones.size();
  std::vector<uint32_t> removeIndex;
  int validCones = 0;
  for(uint32_t i1 = 0; i1 < m_coneFrame.size(); i1++){
    if(m_coneFrame[i1].second.isValid()){
      validCones++;
    }
  }
  if(validCones == 0){
      m_coneFrame.clear();
  }
  for (uint32_t i = 0; i < numberOfCones; i++)
  {
    Cone objectToValidate = cones[i];
    std::pair<bool, Cone> objectPair = std::pair<bool, Cone>(false,objectToValidate);

    if(validCones == 0){
      if(m_verbose)
        std::cout << "in empty frame" << std::endl;
      m_coneFrame.push_back(objectPair);
    }else{
      int k = 0;
      int frameSize = m_coneFrame.size();
      bool foundMatch = false;
      if(m_coneFrame[k].second.isValid()){
        if(objectPair.second.isThisMe(m_coneFrame[k].second.getX(),m_coneFrame[k].second.getY())){
          posShiftX += m_coneFrame[k].second.getX() - objectPair.second.getX();
          posShiftY += m_coneFrame[k].second.getY() - objectPair.second.getY();
          m++;
          m_coneFrame[k].second.setX(objectPair.second.getX());
          m_coneFrame[k].second.setY(objectPair.second.getY());
          m_coneFrame[k].second.addHit();
          m_coneFrame[k].second.addColor(objectPair.second.m_label);
          m_coneFrame[k].first = true;
          foundMatch = true;
        }
      }
      while(k < frameSize && !foundMatch){
        if(!m_coneFrame[k].second.isValid()){
          k++;
          continue;
        }
        if(!m_coneFrame[k].first && objectPair.second.isThisMe(m_coneFrame[k].second.getX(),m_coneFrame[k].second.getY())){
          posShiftX += m_coneFrame[k].second.getX() - objectPair.second.getX();
          posShiftY += m_coneFrame[k].second.getY() - objectPair.second.getY();
          m++;
          m_coneFrame[k].second.setX(objectPair.second.getX());
          m_coneFrame[k].second.setY(objectPair.second.getY());
          m_coneFrame[k].second.addColor(objectPair.second.m_label);
          m_coneFrame[k].second.addHit();
          m_coneFrame[k].first = true;
          foundMatch = true;
        }
        k++;
      }
      if(!foundMatch){
        objectPair.second.addColor(objectPair.second.m_label);
        m_coneFrame.push_back(objectPair);
      }
    }
  }
  std::vector<Cone> sentCones;
  int sentCounter = 0;
  for(uint32_t i = 0; i < m_coneFrame.size(); i++){
    if(m_coneFrame[i].second.isValid()){
      if(m_coneFrame[i].second.shouldBeRemoved()){
        m_coneFrame[i].second.setValidState(false);
        continue;
      }
      if(m_coneFrame[i].first && m_coneFrame[i].second.checkColor()){
        sentCounter++;
        m_coneFrame[i].first = false;
        sentCones.push_back(m_coneFrame[i].second);
      }
      if(!m_coneFrame[i].first){
        m_coneFrame[i].second.addMiss();
      }
    }
  }
  return sentCones;
}

void DetectCone::SendMatchedContainer(std::vector<Cone> cones)
{
  if(cones.size() == 0)
    return;
  cluon::data::TimeStamp sampleTime = m_coneTimeStamp;
  bool noConeDetected = true;
  for(uint32_t n = 0; n < cones.size(); n++){
    if(cones[n].m_label > 0){
      noConeDetected = false;
    }
  }
  if(noConeDetected){
    opendlv::logic::perception::ObjectDirection coneDirection;
    coneDirection.objectId(0);
    coneDirection.azimuthAngle(0);
    coneDirection.zenithAngle(0);
    m_od4.send(coneDirection,sampleTime,m_senderStamp);

    opendlv::logic::perception::ObjectDistance coneDistance;
    coneDistance.objectId(0);
    coneDistance.distance(0);
    m_od4.send(coneDistance,sampleTime,m_senderStamp);

    opendlv::logic::perception::ObjectType coneType;
    coneType.objectId(0);
    coneType.type(666);
    m_od4.send(coneType,sampleTime,m_senderStamp);
    std::cout << "sent 666: " << m_currentFrame << std::endl;

    opendlv::logic::perception::Object object;
    object.objectId(0);
    m_od4.send(object,sampleTime,m_senderStamp);
  }
  else{
    uint32_t numOfOrange = 0;
    for(uint32_t n = 0; n < cones.size(); n++){
      if(cones[n].getLabel() == 4){
        numOfOrange++;
      }
      opendlv::logic::sensation::Point conePoint;
      Cartesian2Spherical(cones[n].getX(), cones[n].getY(), cones[n].getZ(), conePoint);
      // if(m_lidarIsWorking){
      //   LidarToCoG(conePoint);
      // }else{
      //   CameraToCoG(conePoint);
      // }
      LidarToCoG(conePoint);
      if(std::isnan(conePoint.azimuthAngle())||std::isnan(conePoint.zenithAngle())||std::isnan(conePoint.distance())){
        std::cout << "Nan appear! " << m_currentFrame << " " << cones[n].getX() << " " << cones[n].getY() << " " << cones[n].getZ() << " " << conePoint.azimuthAngle() << " " << conePoint.zenithAngle() << " " << conePoint.distance() << " " << cones[n].m_label << std::endl;
      }
      else{
        uint32_t index = cones.size()-1-n;
        opendlv::logic::perception::ObjectDirection coneDirection;
        coneDirection.objectId(index);
        coneDirection.azimuthAngle(-conePoint.azimuthAngle());  //Negative to convert to car frame from LIDAR
        coneDirection.zenithAngle(conePoint.zenithAngle());
        m_od4.send(coneDirection,sampleTime,m_senderStamp);

        opendlv::logic::perception::ObjectDistance coneDistance;
        coneDistance.objectId(index);
        coneDistance.distance(conePoint.distance());
        m_od4.send(coneDistance,sampleTime,m_senderStamp);

        opendlv::logic::perception::ObjectType coneType;
        coneType.objectId(index);
        coneType.type(uint32_t(cones[n].m_label));
        m_od4.send(coneType,sampleTime,m_senderStamp);
      }
    }
    opendlv::logic::perception::Object object;
    object.objectId(numOfOrange);
    m_od4.send(object,sampleTime,m_senderStamp);
  }
}

void DetectCone::saveImages(std::string saveString, cv::Mat img){
  cv::imwrite(saveString,img);
}
