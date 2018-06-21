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
, m_lastLidarData()
, m_lastCameraData()
, m_pointMatched()
, m_diffVec()
, m_finalPointCloud()
, m_threshold(0.9f)
, m_timeDiffMilliseconds()
, m_coneTimeStamp()
, m_imgTimeStamp()
, m_coneCollector()
, m_lastObjectId()
, m_newFrame(true)
, m_processing(false)
, m_coneMutex()
, m_imgMutex()
, m_recievedFirstImg(false)
, m_img()
, m_imgAndTimeStamps()
, m_timeStamps()
, m_currentFrame(0)
, m_offline(0)
, m_annotate(0)
, m_forwardModel()
, m_backwardModel()
, m_lidarIsWorking(false)
, m_checkLidarMilliseconds()
, m_count(0)
, m_patchSize(64)
, m_width(672)
, m_height(376)
, m_xShift(0.1)
, m_yShift(0.9)//0.833
, m_zShift(1.872)
{
  m_diffVec = 0;
  m_pointMatched = Eigen::MatrixXd::Zero(4,1);
  m_lastCameraData = Eigen::MatrixXd::Zero(4,1);
  m_lastLidarData = Eigen::MatrixXd::Zero(4,1);
  m_coneCollector = Eigen::MatrixXd::Zero(4,2000);
  m_lastObjectId = 0;

  setUp(commandlineArguments);
}

DetectCone::~DetectCone()
{
  m_img.release();
  cv::destroyAllWindows();
}

void DetectCone::setUp(std::map<std::string, std::string> commandlineArguments)
{
  m_threshold = static_cast<float>(std::stof(commandlineArguments["threshold"]));
  m_timeDiffMilliseconds = static_cast<int64_t>(std::stoi(commandlineArguments["timeDiffMilliseconds"])); 
  m_checkLidarMilliseconds = static_cast<int64_t>(std::stoi(commandlineArguments["checkLidarMilliseconds"])); 
  m_senderStamp = static_cast<uint32_t>(std::stoi(commandlineArguments["senderStamp"]));
  m_annotate = static_cast<bool>(std::stoi(commandlineArguments["annotate"])); 

  CNN("forward", m_forwardModel);
  CNN("backward", m_backwardModel);
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
    cones(0,coneIndex) = -direction.azimuthAngle();
    cones(1,coneIndex) = direction.zenithAngle();
    cones(2,coneIndex) = distance.distance();
    coneIndex++;
    it++;
  }
  if(cones.cols()>0 && m_recievedFirstImg){
    SendCollectedCones(cones);
  }
}

void DetectCone::nextContainer(cluon::data::Envelope data)
{
  if (data.dataType() == opendlv::logic::perception::ObjectDirection::ID()) {
    m_coneTimeStamp = data.sampleTimeStamp();
    auto coneDirection = cluon::extractMessage<opendlv::logic::perception::ObjectDirection>(std::move(data));
    uint32_t objectId = coneDirection.objectId();
    bool newFrameDist = false;
    {
      std::unique_lock<std::mutex> lockCone(m_coneMutex);
      m_coneCollector(0,objectId) = -coneDirection.azimuthAngle();  //Negative for conversion from car to LIDAR frame
      m_coneCollector(1,objectId) = coneDirection.zenithAngle();
      m_lastObjectId = (m_lastObjectId<objectId)?(objectId):(m_lastObjectId);
      newFrameDist = m_newFrame;
      m_newFrame = false;
    }
    //Check last timestamp if they are from same message
    //std::cout << "Message Recieved " << std::endl;
    if (newFrameDist){
       std::thread coneCollector(&DetectCone::initializeCollection, this);
       coneCollector.detach();
    }
  }

  else if(data.dataType() == opendlv::logic::perception::ObjectDistance::ID()){
    m_coneTimeStamp = data.sampleTimeStamp();;
    auto coneDistance = cluon::extractMessage<opendlv::logic::perception::ObjectDistance>(std::move(data));
    uint32_t objectId = coneDistance.objectId();
    bool newFrameDist = false;
    {
      std::unique_lock<std::mutex> lockCone(m_coneMutex);
      m_coneCollector(2,objectId) = coneDistance.distance();
      m_coneCollector(3,objectId) = 0;
      m_lastObjectId = (m_lastObjectId<objectId)?(objectId):(m_lastObjectId);
      newFrameDist = m_newFrame;
      m_newFrame = false;
    }
    //Check last timestamp if they are from same message
    //std::cout << "Message Recieved " << std::endl;
    if (newFrameDist){
       std::thread coneCollector(&DetectCone::initializeCollection, this);
       coneCollector.detach();
    }
  }
}

void DetectCone::getImgAndTimeStamp(std::pair<cluon::data::TimeStamp, cv::Mat> imgAndTimeStamp){
  if(m_imgAndTimeStamps.size()>20){
    m_imgAndTimeStamps.clear();
  }
  m_imgAndTimeStamps.push_back(imgAndTimeStamp);

  m_imgTimeStamp = imgAndTimeStamp.first;
  m_img = imgAndTimeStamp.second;

  m_recievedFirstImg = true;
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
  else 
    std::cout << "Unable to open timestamp file" << std::endl;

  m_recievedFirstImg = true;
  m_offline = true; 
}

void DetectCone::checkLidarState(){
  if(m_offline){
    if(cluon::time::toMicroseconds(m_coneTimeStamp) == 0){
      std::cout << "No lidar data received" << std::endl;
      m_lidarIsWorking = false;
      if(m_currentFrame < m_timeStamps.size()){
        m_img = cv::imread("/opt/images/"+std::to_string(m_currentFrame++)+".png");
        if(!m_processing){
          m_processing = true;
          forwardDetectionORB(m_img);
          m_processing = false;
        }
        // cv::namedWindow("forwardDetectionORB", cv::WINDOW_NORMAL);
        // cv::imshow("forwardDetectionORB", m_img);
        // cv::waitKey(10);
        // std::cout << m_currentFrame << std::endl;
      }
    }
    else{
      m_count = 0;
      m_lidarIsWorking = true;
    }
  }
  else{
    int64_t timeDiff = cluon::time::toMicroseconds(m_imgTimeStamp) - cluon::time::toMicroseconds(m_coneTimeStamp);
    if ((timeDiff > m_checkLidarMilliseconds*1000)){
      std::cout << "Lidar fails! Camera detection only! " << std::endl;
      m_lidarIsWorking = false;
      if(m_img.empty()){
        return;
      }
      forwardDetectionORB(m_img);
    }
    else{
      m_lidarIsWorking = true;
      // std::cout << "Lidar is working!" << std::endl;
    }
  }
}

void DetectCone::saveRecord(cv::Mat img){
  cv::imwrite("images/"+std::to_string(m_count++)+".png", img);
}

void DetectCone::blockMatching(cv::Mat& disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR, dispL, dispR;

  cv::cvtColor(imgL, grayL, 6);
  cv::cvtColor(imgR, grayR, 6);

  cv::Ptr<cv::StereoBM> sbmL = cv::StereoBM::create(); 
  sbmL->setBlockSize(21);
  sbmL->setNumDisparities(32);
  sbmL->compute(grayL, grayR, dispL);

  disp = dispL/16;
}

void DetectCone::reconstruction(cv::Mat img, cv::Mat& Q, cv::Mat& disp, cv::Mat& rectified, cv::Mat& XYZ){
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

  model << conv(64, 64, 4, 3, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh()                                                   
     << conv(31, 31, 3, 16, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     // << dropout(15*15*16, 0.25)
     << conv(15, 15, 3, 16, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     << conv(7, 7, 3, 32, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     // << dropout(3*3*32, 0.25)                     
     << fc(3 * 3 * 32, 128, true, backend_type) << relu()  
     << fc(128, 5, true, backend_type) << softmax(5); 

  // load nets
  std::ifstream ifs(dictionary.c_str());
  if (!ifs.good()){
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
            cones.push_back(Cone{cv::Point(col, row),maxVal,label});
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


float DetectCone::median(std::vector<float> vec) {
  int size = vec.size();
  float tvecan;
  if (size % 2 == 0) { // even
    tvecan = (vec[vec.size() / 2 - 1] + vec[vec.size() / 2]) / 2;
  }

  else //odd
    tvecan = vec[vec.size() / 2];
  return tvecan;
}

float DetectCone::mean(std::vector<float> vec) {
  float result = 0;
  size_t size = vec.size();
  for(size_t i = 0; i < size; i++){
    result += vec[i];
  }
  result /= size;
  return result;
}

void DetectCone::gather_points(//初始化
  cv::Mat source,
  std::vector<float> vecQuery,
  std::vector<int>& vecIndex,
  std::vector<float>& vecDist
  )
{  
  double radius = 1;
  unsigned int max_neighbours = 100;
  cv::flann::KDTreeIndexParams indexParams(2);
  cv::flann::Index kdtree(source, indexParams); //此部分建立kd-tree索引同上例，故不做详细叙述
  cv::flann::SearchParams params(1024);//设置knnSearch搜索参数
  kdtree.radiusSearch(vecQuery, vecIndex, vecDist, radius, max_neighbours, params);
}

void DetectCone::filterKeypoints(std::vector<cv::Point3f>& point3Ds){
  std::vector<Pt> data;
  
  for(size_t i = 0; i < point3Ds.size(); i++){
    // if(point3Ds[i].y > 0.5 && point3Ds[i].y < 2){
      data.push_back(Pt{point3Ds[i],-1});
    // }
  }
  
  cv::Mat source = cv::Mat(point3Ds).reshape(1);
  point3Ds.clear();
  cv::Point3f point3D;
  int groupId = 0;

  for(size_t j = 0; j < data.size()-1; j++)
  {   
    if(data[j].group == -1){
      std::vector<float> vecQuery(3);//存放 查询点 的容器（本例都是vector类型）
      vecQuery[0] = data[j].pt.x;
      vecQuery[1] = data[j].pt.y;
      vecQuery[2] = data[j].pt.z;
      std::vector<int> vecIndex;
      std::vector<float> vecDist;

      gather_points(source, vecQuery, vecIndex, vecDist);//kd tree finish; find the points in the circle with point center vecQuery and radius, return index in vecIndex
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
        std::vector<float> centerPointX;
        std::vector<float> centerPointY;
        std::vector<float> centerPointZ;
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
              centerPointX.push_back(data[vecIndex[k]].pt.x);
              centerPointY.push_back(data[vecIndex[k]].pt.y);
              centerPointZ.push_back(data[vecIndex[k]].pt.z);
            }
          }
          groupId++;
          point3D.x = mean(centerPointX);
          point3D.y = mean(centerPointY);
          point3D.z = mean(centerPointZ);
        }
        else{
          data[j].group = data[vecIndex[0]].group;
          point3D = data[j].pt;
        }
      }
      if(std::isnan(point3D.x)||std::isnan(point3D.y)||std::isnan(point3D.y))
        continue;
      point3Ds.push_back(point3D);
    }
  }
}

void DetectCone::xyz2xy(cv::Mat Q, cv::Point3f xyz, cv::Point2f& xy, int& radius){
  float X = xyz.x;
  float Y = xyz.y;
  float Z = xyz.z;
  float Cx = float(-Q.at<double>(0,3));
  float Cy = float(-Q.at<double>(1,3));
  float f = float(Q.at<double>(2,3));
  float a = float(Q.at<double>(3,2));
  float b = float(Q.at<double>(3,3));
  float d = (f - Z * b ) / ( Z * a);
  xy.x = X * ( d * a + b ) + Cx;
  xy.y = Y * ( d * a + b ) + Cy;
  radius = int(0.5 * ( d * a + b ));
}

int DetectCone::countFiles(const char* path){
  struct dirent *de;
  DIR *dir = opendir(path);
  if(!dir)
  {
    printf("opendir() failed! Does it exist?\n");
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
  if (0 > roi.x || 0 > roi.width || roi.x + roi.width > img.cols || 0 > roi.y || 0 > roi.height || roi.y + roi.height > img.rows){
    return;
  }
  cv::Mat image = img(roi);
  cv::resize(image, image, cv::Size(m_patchSize,m_patchSize));
  cv::imwrite(path, image);
}

void DetectCone::forwardDetectionORB(cv::Mat img){
  //Given RoI by SIFT detector and detected by CNN
  std::vector<tiny_dnn::tensor_t> inputs;
  std::vector<int> verifiedIndex;
  std::vector<cv::Point> candidates;

  cv::Mat Q, disp, XYZ, imgRoI, imgSource;
  reconstruction(img, Q, disp, img, XYZ);
  img.copyTo(imgSource);

  int rowT = 190;
  int rowB = 320;
  imgRoI = img.rowRange(rowT, rowB);

  cv::Ptr<cv::ORB> detector = cv::ORB::create(100);
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(imgRoI, keypoints);
  if(keypoints.size()==0)
    return;

  cv::Mat probMap[5] = cv::Mat::zeros(m_height, m_width, CV_64F);
  
  std::vector<cv::Point3f> point3Ds;
  cv::Point2f point2D;
  for(size_t i = 0; i < keypoints.size(); i++){
    cv::Point position(int(keypoints[i].pt.x), int(keypoints[i].pt.y)+rowT);
    cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
    if(point3D.y>0.8 && point3D.y<1.1){
      point3Ds.push_back(point3D);
    }
  }
  if(point3Ds.size()==0)
    return;
  filterKeypoints(point3Ds);
  for(size_t i = 0; i < point3Ds.size(); i++){
    int radius;
    xyz2xy(Q, point3Ds[i], point2D, radius);
    int x = int(point2D.x);
    int y = int(point2D.y);

    cv::Rect roi;
    roi.x = std::max(x - radius, 0);
    roi.y = std::max(y - radius, 0);
    roi.width = std::min(x + radius, img.cols) - roi.x;
    roi.height = std::min(y + radius, img.rows) - roi.y;

    if (0 > roi.x || 0 >= roi.width || roi.x + roi.width > img.cols || 0 > roi.y || 0 >= roi.height || roi.y + roi.height > img.rows){
      std::cout << "Wrong roi!" << std::endl;
      continue;
    }
    else{
      auto patchImg = img(roi);
      tiny_dnn::vec_t data;
      convertImage(patchImg, m_patchSize, m_patchSize, data);
      inputs.push_back({data});
      verifiedIndex.push_back(i);
      candidates.push_back(cv::Point(x,y));
    }
  }

  int resultm_width = m_height;
  int resultm_height = m_height;
  double resultResize = 20;
  cv::Mat result = cv::Mat::zeros(resultm_height, resultm_width, CV_8UC3);
  std::string labels[] = {"background", "blue", "yellow", "orange", "big orange"};
  std::vector <Cone> cones;
  // Eigen::MatrixXd cameraCones;
  // int coneCount = 0;

  if(inputs.size()>0){
    auto prob = m_forwardModel.predict(inputs);
    for(size_t i = 0; i < inputs.size(); i++){
      size_t maxIndex = 0;
      double maxProb = prob[i][0][0];
      std::cout << prob[i][0][0] << " ";
      for(size_t j = 1; j < 5; j++){
        std::cout << prob[i][0][j] << " ";
        if(prob[i][0][j] > maxProb){
          maxIndex = j;
          maxProb = prob[i][0][j];
        }
      }
      std::cout << labels[maxIndex] << std::endl;
      int x = candidates[i].x;
      int y = candidates[i].y;
      probMap[maxIndex].at<double>(y,x) = maxProb;
    }
    for(size_t i = 0; i < 5; i++){
      imRegionalMax(cones, i, probMap[i], 10, m_threshold, 20);
    }

    for(size_t i = 0; i < cones.size(); i++){
      int x = cones[i].pt.x;
      int y = cones[i].pt.y;
      double maxProb = cones[i].prob;
      int maxIndex = cones[i].label;
      cv::Point position(x, y);
      cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
      std::string labelName = labels[maxIndex];
      int radius;
      cv::Point2f position_tmp;
      xyz2xy(Q, point3D, position_tmp, radius);

      if(radius<=0){
        continue;
      }

      if(m_annotate){
        annotate(imgSource, maxIndex, position, radius);
      }

      Eigen::MatrixXd cone = Eigen::MatrixXd::Zero(4,1);
      cone << point3D.x-m_xShift, point3D.z-m_zShift, point3D.y-m_yShift, maxIndex;
      // cameraCones.col(coneCount++) = cone;
      if (labelName == "background"){
        std::cout << "No cone detected" << std::endl;
        cv::circle(img, position, radius, cv::Scalar (0,0,0));
      } 
      else{
        if (labelName == "blue")
          cv::circle(img, position, radius, cv::Scalar (255,0,0), 2);
        else if (labelName == "yellow")
          cv::circle(img, position, radius, cv::Scalar (0,255,255), 2);
        else if (labelName == "orange")
          cv::circle(img, position, radius, cv::Scalar (0,165,255), 2);
        else if (labelName == "big orange")
          cv::circle(img, position, radius, cv::Scalar (0,0,255), 2);

        int xt = int(point3D.x * float(resultResize) + resultm_width/2);
        int yt = int(point3D.z * float(resultResize));
        if (xt >= 0 && xt <= resultm_width && yt >= 0 && yt <= resultm_height){
          if (labelName == "blue")
            cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (255,0,0), -1);
          else if (labelName == "yellow")
            cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (0,255,255), -1);
          else if (labelName == "orange")
            cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (0,165,255), -1);
          else if (labelName == "big orange")
            cv::circle(result, cv::Point (xt,yt), 10, cv::Scalar (0,0,255), -1);
        }

        std::cout << position << " " << labelName << " " << point3D << " " << maxProb << std::endl;
      }
    }
  }
      

  for(size_t i = 0; i < keypoints.size(); i++){
    cv::circle(img, cv::Point(int(keypoints[i].pt.x),int(keypoints[i].pt.y)+rowT), 2, cv::Scalar (255,255,255), -1);
  }

  cv::line(img, cv::Point(0,rowT), cv::Point(m_width,rowT), cv::Scalar(0,0,255), 2);
  cv::line(img, cv::Point(0,rowB), cv::Point(m_width,rowB), cv::Scalar(0,0,255), 2);

  // int resultm_width = 672;
  // int resultm_height = 600;
  // double resultResize = 30;
  // cv::Mat result[2] = cv::Mat::zeros(resultm_height, resultm_width, CV_8UC3), coResult;
  // std::string labels[] = {"background", "blue", "yellow", "orange", "big orange"};
  // if(inputs.size()>0){
  //   auto prob = m_CNN.predict(inputs);
  //   for(size_t i = 0; i < inputs.size(); i++){
  //     size_t maxIndex = 1;
  //     double maxProb = prob[i][0][1];
  //     for(size_t j = 2; j < 5; j++){
  //       if(prob[i][0][j] > maxProb){
  //         maxIndex = j;
  //         maxProb = prob[i][0][j];
  //       }
  //     }
  //     // outputs[verifiedIndex[i]] = maxIndex;
  //     int x = candidates[i].x;
  //     int y = candidates[i].y;
  //     cv::Point position(x*2, y*2+180);
  //     cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
  //     std::string labelName = labels[maxIndex];     

  //     if (labelName == "background"){
  //       std::cout << "No cone detected" << std::endl;
  //       cv::circle(img, position, 2, cv::Scalar (0,0,0), -1);
  //     } 
  //     else{
  //       // std::cout << "Find one " << labelName << " cone"<< std::endl;
  //       if (labelName == "blue")
  //         cv::circle(img, position, 2, cv::Scalar (175,238,238), -1);
  //       else if (labelName == "yellow")
  //         cv::circle(img, position, 2, cv::Scalar (0,255,255), -1);
  //       else if (labelName == "orange")
  //         cv::circle(img, position, 2, cv::Scalar (0,165,255), -1);
  //       else if (labelName == "big orange")
  //         cv::circle(img, position, 4, cv::Scalar (0,0,255), -1);

  //       int xt = int(point3D.x * float(resultResize) + resultm_width/2);
  //       int yt = int((point3D.z-1.872f) * float(resultResize));
  //       if (xt >= 0 && xt <= resultm_width && yt >= 0 && yt <= resultm_height){
  //         if (labelName == "blue")
  //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (255,0,0), -1);
  //         else if (labelName == "yellow")
  //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (0,255,255), -1);
  //         else if (labelName == "orange")
  //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (0,165,255), -1);
  //         else if (labelName == "big orange")
  //           cv::circle(result[0], cv::Point (xt,yt), 10, cv::Scalar (0,0,255), -1);
  //       }

  //       std::cout << position << " " << labelName << " " << point3D << std::endl;
  //       savefile << std::to_string(position.x)+","+std::to_string(position.y)+","+labelName+","+std::to_string(point3D.x)+","+std::to_string(point3D.y)+","+std::to_string(point3D.z)+"\n";
  //     }
  //   }
  // }

  // int resultm_width = m_height;
  // int resultm_height = m_height;
  // double resultResize = 20;
  // cv::Mat result = cv::Mat::zeros(resultm_height, resultm_width, CV_8UC3);
  // for(size_t i = 0; i < m_finalPointCloud.cols(); i++){
  //   savefile << std::to_string(m_finalPointCloud(0,i))+","+std::to_string(m_finalPointCloud(1,i))+","+std::to_string(m_finalPointCloud(2,i))+"\n";
  //   int x = int(m_finalPointCloud(0,i) * resultResize + resultm_width/2);
  //   int y = int(m_finalPointCloud(1,i) * resultResize);
  //   if (x >= 0 && x <= resultm_width && y >= 0 && y <= resultm_height){
  //     cv::circle(result[0], cv::Point (x,y), 5, cv::Scalar (255, 255, 255), -1);
  //   }
  // }

  // cv::circle(result[0], cv::Point (int(resultm_width/2),0), 5, cv::Scalar (0, 0, 255), -1);
  // cv::flip(result, result, 0);
  // img.copyTo(result[1].rowRange(resultm_height-376,resultm_height));
  // cv::hconcat(result[1], result[0], coResult);

  // cv::Mat coResult;
  // cv::hconcat(img, result, coResult);
  // cv::namedWindow("disp", cv::WINDOW_NORMAL);
  // cv::imshow("disp", coResult);
  // cv::waitKey(0);
  // cv::waitKey(0);

  // savePath = imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".png";
  // cv::imwrite("result.png", img);

  // savePath = imgPath.substr(0,index-7)+"/disp_filtered/"+filename.substr(0,index2)+".png";
  // std::cout<<savePath<<std::endl;
  // cv::imwrite(savePath, disp);

  // cv::namedWindow("forwardDetectionORB", cv::WINDOW_NORMAL);
  // cv::imshow("forwardDetectionORB", img);
  // cv::waitKey(10);

  cv::imwrite("/opt/results/"+std::to_string(m_currentFrame)+".png", img);

  // if(coneCount){
  //   SendMatchedContainer(cameraCones);
  // }
}

void DetectCone::backwardDetection(cv::Mat img, Eigen::MatrixXd& lidarCones, int64_t minValue){
  //Given RoI in 3D world, project back to the camera frame and then detect
  cv::Mat disp, Q, XYZ, imgSource;
  reconstruction(img, Q, disp, img, XYZ);
  std::vector<tiny_dnn::tensor_t> inputs;
  std::vector<int> verifiedIndex;
  std::vector<cv::Vec3i> porperty;
  img.copyTo(imgSource);

  for(int i = 0; i < lidarCones.cols(); i++){
    cv::Point2f point2D;
    cv::Point3f lidarCone(float(m_xShift+lidarCones(0,i)), float(m_yShift+lidarCones(2,i)), float(m_zShift+lidarCones(1,i)));
    std::cout << lidarCone << std::endl;
    int radius;
    xyz2xy(Q, lidarCone, point2D, radius);

    int x = int(point2D.x);
    int y = int(point2D.y);

    cv::Rect roi;
    roi.x = std::max(x - radius, 0);
    roi.y = std::max(y - radius, 0);
    roi.width = std::min(std::max(x + radius,0), img.cols) - roi.x;
    roi.height = std::min(std::max(y + radius,0), img.rows) - roi.y;

    //cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,0,0));
    // // cv::circle(disp, cv::Point (x,y), 3, 0, CV_FILLED);
    //cv::namedWindow("roi", cv::WINDOW_NORMAL);
    //cv::imshow("roi", img);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
    // std::cout << "RoI for cone " << i << " is  x: " << roi.x << "y: " << roi.y << std::endl;
    if (0 >= roi.x || 0 >= roi.width || roi.x + roi.width > img.cols || 0 >= roi.y || 0 >= roi.height || roi.y + roi.height >= img.rows || radius <= 0){
      std::cout << "Wrong roi!" << std::endl;
      lidarCones(3,i) = -1;
    }
    else{
      auto patchImg = img(roi);
      tiny_dnn::vec_t data;
      convertImage(patchImg, m_patchSize, m_patchSize, data);
      inputs.push_back({data});
      lidarCones(3,i) = 0;
      verifiedIndex.push_back(i);
      porperty.push_back(cv::Vec3i(x,y,radius));
    }
  }

  if(inputs.size()>0){
    auto prob = m_backwardModel.predict(inputs);
    for(size_t i = 0; i < inputs.size(); i++){
      size_t maxIndex = 0;
      float_t maxProb = prob[i][0][0];
      for(size_t j = 1; j < 5; j++){
        if(prob[i][0][j] > maxProb){
          maxIndex = j;
          maxProb = prob[i][0][j];
        }
      }

      lidarCones(3, verifiedIndex[i]) = maxIndex;
      cv::Point position(porperty[i][0],porperty[i][1]);
      int radius = porperty[i][2];

      if(m_annotate){
        annotate(imgSource, maxIndex, position, radius);
      }

      std::string labels[] = {"background", "blue", "yellow", "orange", "big orange"};
      if (maxIndex == 0 || maxProb < m_threshold){
        std::cout << "No cone detected" << std::endl;
        cv::circle(img, position, radius, cv::Scalar (0,0,0));
      } 
      else{
        std::string labelName = labels[maxIndex];
        std::cout << "Find one " << labels[maxIndex-1] << " cone"<< std::endl;
        if (labelName == "blue")
          cv::circle(img, position, radius, cv::Scalar (255,0,0), 2);
        else if (labelName == "yellow")
          cv::circle(img, position, radius, cv::Scalar (0,255,255), 2);
        else if (labelName == "orange")
          cv::circle(img, position, radius, cv::Scalar (0,165,255), 2);
        else if (labelName == "big orange")
          cv::circle(img, position, radius, cv::Scalar (0,0,255), 2);
      }
    }
  }

  //cv::namedWindow("backwardDetection", cv::WINDOW_NORMAL);
  //cv::imshow("backwardDetection", img);
  //cv::waitKey(10);

  cv::imwrite("/opt/results/"+std::to_string(m_currentFrame)+"_"+std::to_string(minValue)+".png", img);
}

Eigen::MatrixXd DetectCone::Spherical2Cartesian(double azimuth, double zenimuth, double distance)
{
  //double xyDistance = distance * cos(azimuth * static_cast<double>(DEG2RAD));
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


void DetectCone::initializeCollection(){
  //std::this_thread::sleep_for(std::chrono::duration 1s); //std::chrono::milliseconds(m_timeDiffMilliseconds)

  bool sleep = true;
  auto start = std::chrono::system_clock::now();

  while(sleep)
  {
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    if ( elapsed.count() > m_timeDiffMilliseconds*1000 )
        sleep = false;
  }


  Eigen::MatrixXd extractedCones;
  {
    std::unique_lock<std::mutex> lockCone(m_coneMutex);
    // std::cout << "FRAME IN LOCK: " << m_newFrame << std::endl;
    extractedCones = m_coneCollector.leftCols(m_lastObjectId+1);
    m_newFrame = true;
    m_lastObjectId = 0;
    m_coneCollector = Eigen::MatrixXd::Zero(4,2000);
  }
  //Initialize for next collection
  //std::cout << "Collection done " << extractedCones.cols() << std::endl;
  if(extractedCones.cols() > 0){
    // std::cout << "Extracted Cones " << std::endl;
    // std::cout << extractedCones << std::endl;
    if(m_recievedFirstImg && !m_processing){
      m_processing = true;
      SendCollectedCones(extractedCones);
      m_processing = false;
    }
  }
}

void DetectCone::SendCollectedCones(Eigen::MatrixXd lidarCones)
{
  //Convert to cartesian
  Eigen::MatrixXd cone;
  for(int p = 0; p < lidarCones.cols(); p++){
    cone = Spherical2Cartesian(lidarCones(0,p), lidarCones(1,p), lidarCones(2,p));
    lidarCones.col(p) = cone;
  }
  if(m_lidarIsWorking){
    std::cout << "Receive lidar data" << std::endl;

    // std::cout << "Start detection" << std::endl;
    //Retrieve Image (Can be extended with timestamp matching)
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
      // std::cout << m_timeStamps[minIndex] << " " << cluon::time::toMicroseconds(m_coneTimeStamp) << " " << minValue << std::endl;
      m_img = cv::imread("/opt/images/"+std::to_string(minIndex)+".png");
      m_currentFrame = minIndex;
    }
    else{
      if (m_imgAndTimeStamps.size() == 0)
        return;
      minIndex = 0;
      minValue = abs(cluon::time::toMicroseconds(m_imgAndTimeStamps[minIndex].first) - cluon::time::toMicroseconds(m_coneTimeStamp));
      for (size_t i = 1; i < m_imgAndTimeStamps.size(); i++){
        int64_t timeDiff = abs(cluon::time::toMicroseconds(m_imgAndTimeStamps[i].first)- cluon::time::toMicroseconds(m_coneTimeStamp));
        if (timeDiff<minValue){
          minIndex = i;
          minValue = timeDiff;
        }
      }
      m_img = m_imgAndTimeStamps[minIndex].second;
    }
    minValue /= 1000;
    if(minValue < 30){
      std::cout << "minIndex: " << minIndex << ", minTimeStampDiff: " << minValue << "ms" << std::endl;
      std::cout << "TimeStamp matched!" << std::endl;  
      backwardDetection(m_img, lidarCones, minValue);
      // cv::Mat rectified = m_img.colRange(0,672);
      // cv::resize(rectified, rectified, cv::Size(672, 376));
      // // rectified.convertTo(rectified, CV_8UC3);
      // // rectified.copyTo(result[1]);
      // // result[1].rowRange(320,680) = rectified;
      // cv::hconcat(result[0], m_img, coResult);
      // cv::imwrite("results/"+std::to_string(m_count++)+".png", coResult);
    }
    else{
      std::cout << "TimeStamp not matched!" << std::endl;
    }
    SendMatchedContainer(lidarCones);
  }
}

void DetectCone::SendMatchedContainer(Eigen::MatrixXd cones)
{
  opendlv::logic::perception::Object object;
  object.objectId(cones.cols());
  std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
  cluon::data::TimeStamp sampleTime = cluon::time::convert(tp);
  m_od4.send(object,sampleTime,m_senderStamp);
  
  for(int n = 0; n < cones.cols(); n++){

    opendlv::logic::sensation::Point conePoint;
    Cartesian2Spherical(cones(0,n), cones(1,n), cones(2,n), conePoint);
    int index = cones.cols()-1-n;
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
    coneType.type(uint32_t(cones(3,n)));
    m_od4.send(coneType,sampleTime,m_senderStamp);
  }
  cluon::data::TimeStamp endTime = cluon::time::now();
  std::cout << "Delta in microseconds" << cluon::time::deltaInMicroseconds(endTime,m_start) << std::endl;
}