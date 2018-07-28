/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "detectcone.hpp"
#include "collector.hpp"

int32_t main(int32_t argc, char **argv) {
    int32_t retCode{0};
    std::map<std::string, std::string> commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if (commandlineArguments.count("cid")<1) {
        std::cerr << argv[0] << " is a detectcone module for the CFSD18 project." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OpenDaVINCI session> [--id=<Identifier in case of simulated units>] [--verbose] [Module specific parameters....]" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=111 --id=120"  <<  std::endl;
        retCode = 1;
    } 
    else {
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
        const uint32_t BPP{static_cast<uint32_t>(std::stoi(commandlineArguments["bpp"]))};
        bool offline{static_cast<bool>(std::stoi(commandlineArguments["offline"]))};
        
        uint32_t attentionSenderStamp = static_cast<uint32_t>(std::stoi(commandlineArguments["attentionSenderStamp"])); 
        uint32_t senderStamp = static_cast<uint32_t>(std::stoi(commandlineArguments["senderStamp"]));
        uint32_t stateMachineStamp = static_cast<uint32_t>(std::stoi(commandlineArguments["stateMachineId"]));

        bool sentReadySignal = false;
        
        cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};
        DetectCone detectcone(commandlineArguments, od4);
        int timeOutMs = std::stoi(commandlineArguments["timeDiffMilliseconds"]);
        int separationTimeMs = std::stoi(commandlineArguments["separationTimeMs"]);
        Collector collector(detectcone,timeOutMs,separationTimeMs,2);

        cluon::data::Envelope data;
        auto envelopeRecieved{[&logic = detectcone, senderStamp = attentionSenderStamp, &collector](cluon::data::Envelope &&envelope)
            {
                if(envelope.senderStamp() == senderStamp){
                    collector.CollectCones(envelope);
                }
            } 
        };

        auto stateMachineStatusEnvelope{[&logic = detectcone, senderStamp = stateMachineStamp](cluon::data::Envelope &&envelope)
        {
            if(envelope.senderStamp() == senderStamp){
                    logic.setStateMachineStatus(envelope);
                }
            }
        };

        od4.dataTrigger(opendlv::logic::perception::ObjectDirection::ID(),envelopeRecieved);
        od4.dataTrigger(opendlv::logic::perception::ObjectDistance::ID(),envelopeRecieved);
        od4.dataTrigger(opendlv::proxy::SwitchStateReading::ID(),stateMachineStatusEnvelope);

        std::stringstream currentDateTime;
        time_t ttNow = time(0);
        tm * ptmNow;
        ptmNow = localtime(&ttNow);
        currentDateTime << 1900 + ptmNow->tm_year << "-";
        if (ptmNow->tm_mon < 9)
            currentDateTime << "0" << 1 + ptmNow->tm_mon << "-";
        else
            currentDateTime << (1 + ptmNow->tm_mon) << "-";
        if (ptmNow->tm_mday < 10)
            currentDateTime << "0" << ptmNow->tm_mday << "_";
        else
            currentDateTime <<  ptmNow->tm_mday << "_";
        if (ptmNow->tm_hour < 10)
            currentDateTime << "0" << ptmNow->tm_hour;
        else
            currentDateTime << ptmNow->tm_hour;
        if (ptmNow->tm_min < 10)
            currentDateTime << "0" << ptmNow->tm_min;
        else
            currentDateTime << ptmNow->tm_min;
        if (ptmNow->tm_sec < 10)
            currentDateTime << "0" << ptmNow->tm_sec;
        else
            currentDateTime << ptmNow->tm_sec;
        std::string folderName = "/opt/"+currentDateTime.str();

        std::string command;

        if(offline){
            std::string fileName = "/opt/replay";
            std::ifstream infile(fileName);
            if(infile.good()){
                command = "rm -r "+fileName;
                system(command.c_str());
            }
            command = "mkdir "+fileName;
            system(command.c_str());
            detectcone.getFolderName(fileName);
            detectcone.getTimeStamp("/opt/timestamp/timestamps.txt");
            while (od4.isRunning()) {
                detectcone.checkLidarState();
                cv::waitKey(1);
            }
        }
        else{
            if ( (BPP != 24) && (BPP != 8) ) {
                std::cerr << argv[0] << ": bits per pixel must be either 24 or 8; found " << BPP << "." << std::endl;
            }
            else {
                const uint32_t SIZE{WIDTH * HEIGHT * BPP/8};
                const std::string NAME{(commandlineArguments["name"].size() != 0) ? commandlineArguments["name"] : "/camera1"};
                const uint32_t ID{(commandlineArguments["id"].size() != 0) ? static_cast<uint32_t>(std::stoi(commandlineArguments["id"])) : 0};

                (void)ID;
                (void)SIZE;

                command = "mkdir "+folderName;
                system(command.c_str());
                command = "mkdir "+folderName+"/timestamp/";
                system(command.c_str());
                command = "mkdir "+folderName+"/images/";
                system(command.c_str());
                command = "mkdir "+folderName+"/results/";
                system(command.c_str());
                detectcone.getFolderName(folderName+"/results");

                std::string filepathTimestamp = folderName+"/timestamp/timestamps.txt";
                std::string imgPath = folderName+"/images/";
                std::ofstream file;
                file.open(filepathTimestamp.c_str());
                size_t frameCounter = 0, readyCounter = 0;
                
                std::unique_ptr<cluon::SharedMemory> sharedMemory(new cluon::SharedMemory{NAME});
                if (sharedMemory && sharedMemory->valid()) {
                    std::clog << argv[0] << ": Found shared memory '" << sharedMemory->name() << "' (" << sharedMemory->size() << " bytes)." << std::endl;

                    cv::Size size;
                    size.width = WIDTH;
                    size.height = HEIGHT;

                    IplImage *image = cvCreateImageHeader(size, IPL_DEPTH_8U, BPP/8);
                    sharedMemory->lock();
                    image->imageData = sharedMemory->data();
                    image->imageDataOrigin = image->imageData;
                    sharedMemory->unlock();
                    bool drivingState = false;
                    while (od4.isRunning()) {
                        // The shared memory uses a pthread broadcast to notify us; just sleep to get awaken up.
                        sharedMemory->wait();
                        
                        sharedMemory->lock();
                        image->imageData = sharedMemory->data();
                        image->imageDataOrigin = image->imageData;
                        cv::Mat img = cv::cvarrToMat(image); 
                        sharedMemory->unlock();
                        cv::waitKey(1);

                        if(readyCounter++ > 150){
                            if(!sentReadySignal){
                                std::cout << "detectcone module is ready!" << std::endl;
                                sentReadySignal = true;
                            }
                            opendlv::system::SignalStatusMessage ssm;
                            ssm.code(1);
                            cluon::data::TimeStamp sampleTime = cluon::time::now();
                            od4.send(ssm, sampleTime, senderStamp);
                        }

                        if(drivingState){
                            cluon::data::TimeStamp imgTimestamp = cluon::time::now();
                            int64_t ts = cluon::time::toMicroseconds(imgTimestamp);
                            std::pair<int64_t, cv::Mat> imgAndTimeStamp(ts, img);
                            detectcone.setTimeStamp(imgAndTimeStamp);
                            detectcone.checkLidarState();
                            file << std::setprecision(19) << ts << std::endl;
                            std::string saveString = imgPath + std::to_string(frameCounter++) + ".png";
                            std::thread imWriteThread(&DetectCone::saveImages,&detectcone,saveString,img);
                            imWriteThread.detach();
                        }else{
                            drivingState = detectcone.getdrivingState();
                        }                       
                    }
                    file.close();
                    cvReleaseImageHeader(&image);
                }
                else {
                    std::cerr << argv[0] << ": Failed to access shared memory '" << NAME << "', camera fails!" << std::endl;
                    opendlv::system::SignalStatusMessage ssm;
                    ssm.code(0);
                    cluon::data::TimeStamp sampleTime = cluon::time::now();
                    od4.send(ssm, sampleTime, senderStamp);
                }
            }
        
        }
    }
    return retCode;
}








