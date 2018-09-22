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

#ifndef CONE_HPP
#define CONE_HPP

#include <opencv2/opencv.hpp>

class Cone{
  public:
    Cone(double x,double y,double z);
    ~Cone() = default;

    cv::Point m_pt;
    double m_prob;
    size_t m_label;
    double getX();
    double getY();
    double getZ();
    size_t getLabel();
    void setX(double x);
    void setY(double y);
    void setZ(double z);
    void addHit();
    int getHits();
    void addMiss();
    int getMisses();
    bool isThisMe(double x, double y);
    bool shouldBeInFrame();
    bool shouldBeRemoved();
    void setValidState(bool state);
    bool isValid();
    bool checkColor();
    void addColor(size_t label);

  private:
    double m_x;
    double m_y;
    double m_z;
    int m_hits = 0;
    int m_missHit = 0;
    bool m_isValid = true;
    std::vector<uint32_t> m_colorList = {};
    int m_noDetectionCount = 0;
    int m_blueCount = 0;
    int m_yellowCount = 0;
    int m_orangeCount = 0;
};

#endif

