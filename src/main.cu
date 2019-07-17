#include <iostream>
#include "videoInput.h"
#include "wisenetMD.h"



#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;


static const char* about =
"This sample uses You only look once (YOLO)-Detector (https://arxiv.org/abs/1612.08242) to detect objects on camera/video/image.\n"
"Models can be downloaded here: https://pjreddie.com/darknet/yolo/\n"
"Default network is 416x416.\n"
"Class names can be downloaded here: https://github.com/pjreddie/darknet/tree/master/data\n";



static const char* params =
"{ help           | false | print usage         }"
"{ cfg            |       | model configuration }"
"{ model          |       | model weights       }"
"{ camera_device  | 0     | camera device number}"
"{ source         |       | video or image for detection}"
"{ out            |       | path to output video file}"
"{ fps            | 3     | frame per second }"
"{ style          | box   | box or line style draw }"
"{ min_confidence | 0.24  | min confidence      }"
"{ class_names    |       | File with class names, [PATH-TO-DARKNET]/data/coco.names }";


int main(int argc, char** argv){
    
    // std::string path = "/home/texs/Documents/dataset2014/dataset/dynamicBackground/fall/input/in%06d.jpg";
    std::string path = "/home/texs/Documents/dataset2014/dataset/baseline/pedestrians/input/in%06d.jpg";
    VideoInput input(path);
    input.resize(cv::Size(300, 200));
    
    WisenetMD wn;

    wn.set_input(&input);
    input.grab_frame();
    wn.init();
    wn.init_model();

    cv::Mat img;
    cv::Mat img_text;



    CommandLineParser parser(argc, argv, params);
    
    String modelConfiguration = parser.get<String>("cfg");
    String modelBinary = parser.get<String>("model");
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    
    // VideoCapture cap;
    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    double fps = parser.get<float>("fps");
    
    // int cameraDevice = parser.get<int>("camera_device");
    // cap = VideoCapture(cameraDevice);
    
   
    vector<String> classNamesVec;
    ifstream classNamesFile(parser.get<String>("class_names").c_str());
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
    String object_roi_style = parser.get<String>("style");





    

    while (true)
    // while (!input.is_empty())
    {
        Mat frame;
        input.grab_frame();
        wn.process_frame();
        devs();

        input.get_mat(frame);
        // input.get_texture_mat(img_text);
        
        cv::imshow("video", frame);
        // cv::imshow("texture", img_text);

        // - ---------------YOLO
        // cap >> frame; // get a new frame from camera/video or read image
        // cv::resize(frame, frame, cv::Size(300, 200));
        if(wn.get_movement_ratio() >= 0.001){
            input.get_mat(frame);
            input.get_mat(frame);

            Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
            net.setInput(inputBlob, "data");                   //set the network input
            Mat detectionMat = net.forward("detection_out");   //compute output
            vector<double> layersTimings;
            double tick_freq = getTickFrequency();
            double time_ms = net.getPerfProfile(layersTimings) / tick_freq * 1000;
            putText(frame, format("FPS: %.2f ; time: %.2f ms", 1000.f / time_ms, time_ms),
                    Point(20, 20), 0, 0.5, Scalar(0, 0, 255));
            float confidenceThreshold = parser.get<float>("min_confidence");
            for (int i = 0; i < detectionMat.rows; i++)
            {
                const int probability_index = 5;
                const int probability_size = detectionMat.cols - probability_index;
                float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
                size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
                float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
                if (confidence > confidenceThreshold)
                {
                    float x_center = detectionMat.at<float>(i, 0) * frame.cols;
                    float y_center = detectionMat.at<float>(i, 1) * frame.rows;
                    float width = detectionMat.at<float>(i, 2) * frame.cols;
                    float height = detectionMat.at<float>(i, 3) * frame.rows;
                    Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                    Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                    Rect object(p1, p2);
                    Scalar object_roi_color(0, 255, 0);
                    if (object_roi_style == "box")
                    {
                        rectangle(frame, object, object_roi_color);
                    }
                    else
                    {
                        Point p_center(cvRound(x_center), cvRound(y_center));
                        line(frame, object.tl(), p_center, object_roi_color, 1);
                    }
                    String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);
                    String label = format("%s: %.2f", className.c_str(), confidence);
                    int baseLine = 0;
                    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    rectangle(frame, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)),
                              object_roi_color, FILLED);
                    putText(frame, label, p1 + Point(0, labelSize.height),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
                }
            }
            
            
        }
        imshow("YOLO: Detections", frame);

        
        cv::waitKey(10);

        
        
    }
    

    return 0;
}





















// int main(){
//     std::string path = "/home/texs/Documents/dataset2014/dataset/baseline/pedestrians/input/in%06d.jpg";
//     VideoInput input(path);
//     std::cout<<input.get_count()<<std::endl;

//     cv::Mat img;
//     cv::Mat img_text;

//     utils::Clock reloj_h;
//     cudu::Clock reloj_d;
//     while (!input.is_empty())
//     {
//         reloj_h.start();
//         reloj_d.start();

//         input.grab_frame();
//         input.get_mat(img);
//         input.get_texture_mat(img_text);
        
//         reloj_d.stop();
//         cudaDeviceSynchronize();
//         cv::imshow("video", img);
//         cv::imshow("texture", img_text);
//         cv::waitKey(10);
//         reloj_h.stop();
//         std::cout<<"host: "<<reloj_h.time()<<"ms  device: "<<reloj_d.time()<<"ms"<<std::endl;
//     }
    

//     return 0;
// }