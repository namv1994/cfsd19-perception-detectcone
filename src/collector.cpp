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

#include "collector.hpp"

Collector::Collector(DetectCone &detectcone,int timeOutMs, int separationTimeMs, int packetSize) :
    m_module(detectcone),
    m_packetSize(packetSize),
    m_timeOutMs(timeOutMs),
    m_separationTimeMs(separationTimeMs),
    m_timeReceived()
{
}

void Collector::CollectCones(cluon::data::Envelope data){
    cluon::data::TimeStamp ts = data.sampleTimeStamp();
    int64_t delta = cluon::time::deltaInMicroseconds(ts,m_currentFrameTime);
    m_timeReceived = std::chrono::system_clock::now();
    if(std::abs(delta)<1){
        if(data.dataType() == opendlv::logic::perception::ObjectDirection::ID()){
            opendlv::logic::perception::ObjectDirection direction = cluon::extractMessage<opendlv::logic::perception::ObjectDirection>(std::move(data));
            uint32_t id = direction.objectId();
            std::map<int,ConePackage>::iterator it;
            it = m_currentFrame.find(id);
            if(it!=m_currentFrame.end()){
                std::get<0>(it->second) = direction;
                m_envelopeCount[id]++;
            }
            else{
                ConePackage conePacket;
                std::get<0>(conePacket) = direction;
                m_currentFrame[id] = conePacket;
                m_envelopeCount[id] = 1;
            }
            m_numberOfItems = (m_numberOfItems<=id)?(id+1):(m_numberOfItems);
        }
        else if(data.dataType() == opendlv::logic::perception::ObjectDistance::ID()){
            opendlv::logic::perception::ObjectDistance distance = cluon::extractMessage<opendlv::logic::perception::ObjectDistance>(std::move(data));
            uint32_t id = distance.objectId();
            std::map<int,ConePackage>::iterator it;
            it = m_currentFrame.find(id);
            if(it!=m_currentFrame.end()){
                std::get<1>(it->second) = distance;
                m_envelopeCount[id]++;
            }
            else{
                ConePackage conePacket;
                std::get<1>(conePacket) = distance;
                m_currentFrame[id] = conePacket;
                m_envelopeCount[id]=1;
            }
            m_numberOfItems = (m_numberOfItems<=id)?(id+1):(m_numberOfItems);
        }
        else if(data.dataType() == opendlv::logic::perception::ObjectType::ID()){
            opendlv::logic::perception::ObjectType type = cluon::extractMessage<opendlv::logic::perception::ObjectType>(std::move(data));
            uint32_t id = type.objectId();
            std::map<int,ConePackage>::iterator it;
            it = m_currentFrame.find(id);
            if(it!=m_currentFrame.end()){
                std::get<2>(it->second) = type;
                m_envelopeCount[id]++;
            }
            else{
                ConePackage conePacket;
                std::get<2>(conePacket) = type;
                m_currentFrame[id] = conePacket;
                m_envelopeCount[id]=1;
            }
            m_numberOfItems = (m_numberOfItems<=id)?(id+1):(m_numberOfItems);
        }
        m_messageCount++;

    }
    else if(m_newFrame)
    {
        m_numberOfItems = 1;
        m_currentFrame.clear();
        m_envelopeCount.clear();
        m_messageCount = 1;
        m_currentFrameTime = data.sampleTimeStamp();
        m_newFrame = false;
        if(data.dataType() == opendlv::logic::perception::ObjectDirection::ID()){
            opendlv::logic::perception::ObjectDirection direction = cluon::extractMessage<opendlv::logic::perception::ObjectDirection>(std::move(data));
            uint32_t id = direction.objectId();
            ConePackage conePacket;
            std::get<0>(conePacket) = direction;
            m_currentFrame[id] = conePacket;
            m_envelopeCount[id]=1;
            m_numberOfItems = (m_numberOfItems<=id)?(id+1):(m_numberOfItems);
        }
        else if(data.dataType() == opendlv::logic::perception::ObjectDistance::ID()){
            opendlv::logic::perception::ObjectDistance distance = cluon::extractMessage<opendlv::logic::perception::ObjectDistance>(std::move(data));
            uint32_t id = distance.objectId();
            ConePackage conePacket;
            std::get<1>(conePacket) = distance;
            m_currentFrame[id] = conePacket;
            m_envelopeCount[id]=1;
            m_numberOfItems = (m_numberOfItems<=id)?(id+1):(m_numberOfItems);
        }
        else if(data.dataType() == opendlv::logic::perception::ObjectType::ID()){
            opendlv::logic::perception::ObjectType type = cluon::extractMessage<opendlv::logic::perception::ObjectType>(std::move(data));
            uint32_t id = type.objectId();
            ConePackage conePacket;
            std::get<2>(conePacket) = type;
            m_currentFrame[id] = conePacket;
            m_envelopeCount[id]=1;
            m_numberOfItems = (m_numberOfItems<=id)?(id+1):(m_numberOfItems);
        }
        std::thread coneCollector (&Collector::InitializeCollection,this); //just sleep instead maybe since this is unclear how it works
        coneCollector.detach();
    }
    else{
        std::cout << "Leaking frames wtf!!!" << std::endl;
    }

}

void Collector::InitializeCollection(){
bool sleep = true;
auto start = std::chrono::system_clock::now();

  while(sleep)
  {
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    std::chrono::duration<double> dur = now-m_timeReceived;
    if(m_messageCount == m_numberOfItems*m_packetSize){
        sleep = false;
    }
    if(elapsed.count() > m_timeOutMs*1000){
        std::cout << "Timed out" << std::endl;
        sleep = false;
    }
    if (dur.count()>m_separationTimeMs*0.001) {
      std::cout << "Separation time exceeded: " <<" dur.count(): "<<dur.count()<<" m_separationTimeMs*0.001 "<<m_separationTimeMs*0.001<<std::endl;
      sleep = false;
    }
  }
  GetCompleteFrame();
  SendFrame();
  m_newFrame = true;
}

void Collector::GetCompleteFrame(){
    std::map<int,int>::iterator it2 = m_envelopeCount.begin();
    std::map<int,ConePackage> currentFrameCopy = m_currentFrame;
    while(it2 != m_envelopeCount.end()){
        if(it2->second != static_cast<int>(m_packetSize)){
            currentFrameCopy.erase(it2->first);
            std::cout << "Incomplete frame with id " << it2->first << " removed" << std::endl;
        }
        it2++;
        //it++;
    }
    if (m_currentFrame.size() != currentFrameCopy.size()) {
      m_currentFrame = currentFrameCopy;
    }
}

void Collector::SendFrame(){
    std::cout << "sending " << m_currentFrame.size() << " cones" << std::endl;
    m_module.receiveCombinedMessage(m_currentFrameTime,m_currentFrame);
}
