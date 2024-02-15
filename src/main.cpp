#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "person_segmentator/person_segmentator.h"

using namespace cv;
using namespace std;

#define WIDTH 224
#define HEIGHT 224
#define FPS 24

int main()
{
    // the setup of the camera feed
    const string window_name = "Camera feed";
    auto modelPath = "/Users/radiakbar/Projects/ort_av/assets/LRASPP.onnx";

    // fps variables
    double start_time, current_time;
    double fps;

    // Create segmentator
    PersonSegmentator ps(modelPath);

    // load the videocapture
    VideoCapture cap = VideoCapture(0);
    cap.set(CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
    cap.set(CAP_PROP_FPS, FPS);
    Mat frame, maskedImage, imageMask;

    // if you can't open the video camera, then print error
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video camera" << endl;
        cin.get();
    }
    // setup the window name
    namedWindow(window_name);

    while (cap.isOpened())
    {
        start_time = static_cast<double>(getTickCount());

        // read the frame
        cap.read(frame);
        resize(frame, frame, Size(HEIGHT, WIDTH), INTER_LINEAR); 

        // Input the frame
        imageMask = ps.Inference(frame);

        // Edit the image
        addWeighted(imageMask, 1, frame, 1, 0, maskedImage);

        // compute the frames per second
        current_time = static_cast<double>(getTickCount());
        fps = 1 / ((current_time - start_time) / getTickFrequency());

        // display fps
        cv::putText(maskedImage, "FPS: " + to_string(fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // Display the image
        imshow("Semantic Segmentation Predictions", maskedImage);

        // wait for 10 ms and the esc key to break loop
        if (waitKey(10) == 27)
        {
            cout << "Esc key is pressed by user. Stopping video" << endl;
            break;
        }

    }

    return 0;
}