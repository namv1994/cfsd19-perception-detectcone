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

#ifndef CFSD18_PERCEPTION_DETECTCONE_HPP
#define CFSD18_PERCEPTION_DETECTCONE_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <thread>
#include <Eigen/Dense>
#include <cstdint>
#include <tuple>
#include <utility>
#include <string>
#include <sstream>

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tiny_dnn/tiny_dnn.h>

#include "cone.hpp"
#include "point.hpp"


class DetectCone {
 public:
  DetectCone(std::map<std::string, std::string> commandlineArguments, cluon::OD4Session &od4);
  DetectCone(DetectCone const &) = delete;
  DetectCone &operator=(DetectCone const &) = delete;
  ~DetectCone();
  void nextContainer(cluon::data::Envelope data);
  void checkLidarState();
  void getImgAndTimeStamp(std::pair<cluon::data::TimeStamp, cv::Mat>);
  void getTimeStamp(const std::string);


 private:
  void setUp(std::map<std::string, std::string> commandlineArguments); 
  void saveRecord(cv::Mat);
  void blockMatching(cv::Mat&, cv::Mat, cv::Mat);
  void reconstruction(cv::Mat, cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&);
  void convertImage(cv::Mat, int, int, tiny_dnn::vec_t&);
  void CNN(const std::string&);
  void imRegionalMax(std::vector<Cone>&, size_t, cv::Mat, int, double, int);
  float median(std::vector<float>);
  float mean(std::vector<float>);
  void gather_points(cv::Mat, std::vector<float>, std::vector<int>&, std::vector<float>&);
  void filterKeypoints(std::vector<cv::Point3f>&);
  void xyz2xy(cv::Mat, cv::Point3f, cv::Point2f&, int&);
  void forwardDetectionORB(cv::Mat);
  void backwardDetection(cv::Mat, std::vector<cv::Point3f>, std::vector<int>&);

  void initializeCollection();
  Eigen::MatrixXd Spherical2Cartesian(double, double, double);
  void Cartesian2Spherical(double, double, double, opendlv::logic::sensation::Point&);
  void SendCollectedCones(Eigen::MatrixXd);
  void SendMatchedContainer(Eigen::MatrixXd);

  cluon::OD4Session &m_od4;
  Eigen::MatrixXd m_lastLidarData;
  Eigen::MatrixXd m_lastCameraData;
  Eigen::MatrixXd m_pointMatched;
  double m_diffVec;
  Eigen::MatrixXd m_finalPointCloud;
  double m_threshold;
  int64_t m_timeDiffMilliseconds;
  cluon::data::TimeStamp m_coneTimeStamp;
  cluon::data::TimeStamp m_imgTimeStamp;
  Eigen::MatrixXd m_coneCollector;
  uint32_t m_lastObjectId;
  bool m_newFrame;
  bool m_processing;
  std::mutex m_coneMutex;
  std::mutex m_imgMutex;
  bool m_recievedFirstImg;
  cv::Mat m_img;
  std::vector<std::pair<cluon::data::TimeStamp, cv::Mat>> m_imgAndTimeStamps;
  std::vector<int64_t> m_timeStamps;
  uint32_t m_currentFrame;
  bool m_offline;
  tiny_dnn::network<tiny_dnn::sequential> m_CNN;
  bool m_lidarIsWorking;
  int64_t m_checkLidarMilliseconds;
  uint32_t m_senderStamp = 118;
  uint32_t m_attentionSenderStamp = 116;
  uint32_t m_count;
  int m_patchSize;
  int m_width;
  int m_height;
  double m_xShift;                //lateral distance between camera and LiDAR 
  double m_yShift;            //Height difference between camera and LiDAR
  double m_zShift;        //Distance between camera and LiDAR in forward distance

  const double DEG2RAD = 0.017453292522222; // PI/180.0
  const double RAD2DEG = 57.295779513082325; // 1.0 / DEG2RAD;
};


#endif