//
//  dataset.cpp
//
//  Created by Terna Kpamber on 2/9/16.
//  Copyright © 2016 Terna Kpamber. All rights reserved.
//

#include <iostream>
using namespace std;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
using namespace cv;

int main(int argc, const char * argv[]) {
	
	string fileName = "folder";
	string ext = ".png";
	Mat frame, roi;
	Point origin = Point(300, 100);
	Point end = Point(240, 150);

	//initialize and set camera attributes
	VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_HEIGHT, 320);
	cap.set(CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CAP_PROP_FORMAT, CV_32F);
	cap.set(CAP_PROP_FPS, 30);
	namedWindow("frame");
	
	cout << "Recording video @ 30Fps 320X320" << endl;
	cout << "THE SUQARE WILL TURN GREEN TO INDICATE FRAMES ARE BEING CAPTURED" << endl;
	
	int numFrames = 500;
	int count = 0;
	int setup = 0;

	//give time to get into gesture position
	while(setup < 100){
		if(cap.grab()){
			setup++;
			cap.retrieve(frame);
			flip(frame, frame, 1);
			rectangle(frame, origin, end, Scalar(0, 0, 255));
			imshow("frame", frame);
		}
	}
	
	//start capturing frames
	//flip the frame horizontally
	//then resize from 320x320 to 32x32
	while(count < numFrames){
		if(cap.grab()){
			count++;
			cap.retrieve(frame);
			flip(frame, frame, 1);
			imshow("frame", frame);
			resize(frame, frame, Size(32,32));
			imwrite(fileName+to_string(count)+ext, frame);
		}
	}
	
	cap.release();
	
}