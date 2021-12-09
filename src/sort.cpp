#include <sstream>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "include/run_yolo.h"
#include "include/hungarian.h"
#include <string>


#include <opencv2/tracking/kalman_filters.hpp>
using namespace std;
static cv::Mat frame;

static cv::String weightpath ="/home/wonder/camera/src/include/yolov4-tiny.weights";
static cv::String cfgpath ="/home/wonder/camera/src/include/yolov4-tiny.cfg";
static cv::String classnamepath = "/home/wonder/camera/src/include/coco.names";

static run_yolo Yolonet(cfgpath, weightpath, classnamepath, float(0.85));

int main(int argc, char** argv)
{

    cout<<"Object detection..."<<endl;



    // >>>> Kalman Filter
    int stateSize = 4;
    int measSize = 2;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y]
    cv::Mat meas(measSize, 1, type);    // [z_x,z_y]
    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  ]
    // [ 0 1 0  dT ]
    // [ 0 0 1  0  ]
    // [ 0 0 0  1  ]

    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 ]
    // [ 0 1 0 0 ]

    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(5) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0    ]
    // [ 0    Ey  0     0    ]
    // [ 0    0   Ev_x  0    ]
    // [ 0    0   0     Ev_y ]

    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(5) = 1e-2;
    kf.processNoiseCov.at<float>(10) = 5.0f;
    kf.processNoiseCov.at<float>(15) = 5.0f;


    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // <<<< Kalman Filter

    // Camera Index
    int idx = 0;

    // Camera Capture
    cv::VideoCapture cap("/home/wonder/camera/src/include/test.mp4");
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    cv::VideoWriter video("/home/wonder/camera/src/include/tracking.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(960,540));
    //cv::VideoWriter videoyolo("/home/patrick/catkin_ws/src/offb/src/include/yolo/yolo.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(640,480));

    bool found = false, got = false;
    int notFoundCount = 0;
    cv::Mat frame;
    double dT;
    double tpf=0;
    int w = 200,h = 200;
    double ticks = 0;


    vector<MOT> previous;
    vector<MOT> detected;
    vector<MOT> objstates;
    vector<cv::Mat> measurements;
    hungarian assign;
    vector<objectinfo> objects;
    vector<int> erase_index;
    vector<int> new_candidate;
    int id = 0, track_id = 1, missing = 0;
    while(1)
    {
        cap >> frame;
        cv::resize(frame,frame,cv::Size(0.5*frame.cols,0.5*frame.rows));
//        cout<<frame.size()<<endl;
        cv::Mat res;
        frame.copyTo( res );
        double precTick = ticks;
        ticks = (double) cv::getTickCount();

        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
//        cout<<dT<<endl;
        if (found)
        {
            // >>>> Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(7) = dT;
            // <<<< Matrix A
//            cout << "dT:" << endl << dT << endl;
//            cout << "State post:" << endl << state << endl;
//            cout<<"now size"<<objstates.size()<<endl;
            for(int i = 0;i<objstates.size();i++)
            {
                kf.statePost = objstates[i].state_of_1obj.clone(); // X'(k-1)
                kf.predict();
                objstates[i].state_of_1obj = kf.statePre.clone(); // X'(-k)
//                cout<<objstates[i].state_of_1obj<<endl;
                cv::circle(res,cv::Point(objstates[i].state_of_1obj.at<float>(0), objstates[i].state_of_1obj.at<float>(1)),4,CV_RGB(255,255,0),-1);
                string idonframe = to_string(objstates[i].id);
                cv::putText(res,idonframe, cv::Point(objstates[i].state_of_1obj.at<float>(0)+10, objstates[i].state_of_1obj.at<float>(1)+10),cv::FONT_HERSHEY_COMPLEX_SMALL,1,CV_RGB(255,0,0));
            }

        }
        Yolonet.rundarknet(frame);
        Yolonet.display(frame);

        if(Yolonet.obj_vector.size()>0)
            got=true;



        if(!got)
        {
//            notFoundCount++;
//            cout << "notFoundCount:" << notFoundCount << endl;
//            if(notFoundCount>100)
//            {
//                found = false;
//            }
        }
        else
        {
            measurements.clear();
            for(auto o:Yolonet.obj_vector)
            {
                if(o.classnameofdetection == "person")
                {
                    meas.at<float>(0) = o.boundingbox.x + o.boundingbox.width/2;
                    meas.at<float>(1) = o.boundingbox.y + o.boundingbox.height/2;
                    meas.at<float>(2) = o.boundingbox.width;
                    meas.at<float>(3) = o.boundingbox.height;
                    cv::Mat temp = meas;
                    measurements.push_back(meas.clone());//save every state of object into an array
                }
            }
            cout<<"Zk size:"<<measurements.size()<<endl;
            cout<<"state size"<<objstates.size()<<endl;
            //all the measurements (Zk) are here
            //update every frame

            notFoundCount = 0;            


            if (!found) // First detection!
            {
                // >>>> Initialization the first measurements

                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(5) = 1; // px
                kf.errorCovPre.at<float>(10) = 1;
                kf.errorCovPre.at<float>(15) = 1;
                //the P matrix

                for(int i=0;i<measurements.size();i++)
                {
                    //save every first measured boxes
                    //also assgin id to them
                    state.at<float>(0) = measurements[i].at<float>(0);
                    state.at<float>(1) = measurements[i].at<float>(1);
                    state.at<float>(2) = 0;
                    state.at<float>(3) = 0;
                    MOT temp = {state.clone(), id, 0, missing};
                    objstates.push_back(temp);
                    id++;
                }

                // <<<< Initialization
                found = true;
            }
            else
            {
                vector<Match> ids = assign.solution(measurements, objstates);   //Munkres Algo Hungarian Algo
                //detection ids: correspond to which object state
//                cout<<id
                for(int i = 0;i<ids.size();i++)
                {
                    cout<<i<<endl;
                    if(ids[i].toofar /*&& i < objstates.size() && i < measurements.size()*/)
                    {
                        //reject & ->
                        new_candidate.push_back(i);
                    }
                    else
                    {
                        cout<<"matched"<<endl;
                        if(ids[i].id >= objstates.size())
                        {
                            new_candidate.push_back(i);
                        }
                        else
                        {
                            if(i>=measurements.size())
                                continue;
                            kf.statePre = objstates[ids[i].id].state_of_1obj;
                            cout<<1<<endl;
                            cout<<measurements[i]<<endl;
                            kf.correct(measurements[i]);
                            cout<<2<<endl;
                            objstates[ids[i].id].state_of_1obj = kf.statePost.clone();
                            cout<<4<<endl;
                            objstates[ids[i].id].tracked_i = 1;
                            objstates[ids[i].id].missing = 0;
                        }
                    }
                }


                for (int i=0;i<objstates.size();i++)
                {
                    if(objstates[i].tracked_i == 0)
                    {
                        //meaning that it was missed
                        cout<<"missed"<<endl;
                        if(objstates[i].missing >= 40 )//remove after lossed for too long
                        {
                            cout<<"erase"<<endl;
                            erase_index.push_back(i);
                        }
                        else
                        {
                            kf.statePre = objstates[i].state_of_1obj;
                            meas.at<float>(0) = objstates[i].state_of_1obj.at<float>(0);
                            meas.at<float>(1) = objstates[i].state_of_1obj.at<float>(1);
                            kf.correct(meas);
                            objstates[i].state_of_1obj = kf.statePost.clone();
                            //objstates[i].tracked_i++;
                            objstates[i].missing += 1;
                        }
                    }

                }

                for (int i=0;i<new_candidate.size();i++)
                {
                    state.at<float>(0) = measurements[new_candidate[i]].at<float>(0);
                    state.at<float>(1) = measurements[new_candidate[i]].at<float>(1);
                    state.at<float>(2) = 0;
                    state.at<float>(3) = 0;
                    missing = 0;
                    MOT temp = {state.clone(), id, track_id, missing};
                    objstates.push_back(temp);
                    id++;
                }


                for (int i=0;i<objstates.size();i++)
                {
                    objstates[i].tracked_i = 0;
                }
                new_candidate.clear();
                // Kalman Correction
            }
        }

        int count = 0;
        sort(erase_index.begin(), erase_index.end());  // Make sure the container is sorted
        for(int e = 0; e<erase_index.size();e++)
        {
            objstates.erase(objstates.begin() + erase_index[e] - count);
        }
        erase_index.clear();
//        sort(erase_index.begin(), erase_index.end());  // Make sure the container is sorted
//        for (auto &i = erase_index.rbegin(); i != erase_index.rend(); ++ i)
//        {
//            objstates.erase(objstates.begin() + *i);
//        }
        cv::imshow("Tracking", res);
        cv::waitKey(20);
        video.write(res);




        if(frame.empty())
            break;

        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;

    }

    return 0;
}

    // [


