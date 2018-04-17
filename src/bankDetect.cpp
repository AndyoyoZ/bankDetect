
#include <algorithm>
#include "bankDetect.h"

//-----------------on_Trackbar( )-------------------//
//--------------------------------------------------//
//-------------------滑动条-------------------------//
void IPSG::CbankDetect::on_Trackbar(int, void*)
{
	iLowL = cv::getTrackbarPos("LowL", "Control");
	iHighL = cv::getTrackbarPos("HighL", "Control");

	iLowA = cv::getTrackbarPos("LowA", "Control");
	iHighA = cv::getTrackbarPos("HighA", "Control");

	iLowB = cv::getTrackbarPos("LowB", "Control");
	iHighB = cv::getTrackbarPos("HighB", "Control");
}

//-----------------getPoint( )----------------------
//--------------------------------------------------
//            得到水岸线离散点集
void IPSG::CbankDetect::getPoint(cv::Mat &img, std::vector<cv::Point> &inputPoint)
{
	int i, j;
	int height = img.rows;
	int width = img.cols;
	for (i = 0; i < width; i++)
	{
		cv::Point tempPoint = cv::Point(0, 0);
		for (j = 0; j < height; j++)
		{
			if (0 != img.at<uchar>(j, i))
			{
				tempPoint = cv::Point(i, j);
			}
		}
		if (cv::Point(0, 0) != tempPoint)
		{
			inputPoint.push_back(tempPoint);
		}
	}
}

//-----------------drawExtendLine( )----------------------
//--------------------------------------------------------
//                   画延长线
void IPSG::CbankDetect::drawExtendLine(cv::Mat &img, cv::Point pt1, cv::Point pt2, cv::Point pt3, cv::Point pt4, cv::Scalar color, int thickness, int line_type)
{
	double k = (pt2.y - pt1.y) / (pt2.x - pt1.x + 0.000001);//斜率可能为0，因此加0.000001
	double h = img.rows;
	double w = img.cols;

	pt3.x = pt1.x - pt1.y / k;
	pt3.y = 0;
	pt4.x = 0;
	pt4.y = pt2.y - k * pt2.x;
	cv::line(img, pt3, pt4, cv::Scalar(0, 255, 255), 2, 8);
}

//-----------------segment( )----------------------
//--------------------------------------------------
//                  图像分割
void IPSG::CbankDetect::segment(cv::Mat &inputImg, cv::Point pt1, cv::Point pt2, cv::Mat &outputImg)
{
	cv::Mat mask = cv::Mat::zeros(inputImg.size(), CV_8UC1);
	cv::Point2d pt3, pt4;
	double k = (pt2.y - pt1.y) / (pt2.x - pt1.x + 0.000001);//斜率可能为0，因此加0.000001
	double h = inputImg.rows;
	double w = inputImg.cols;
	//计算端点
	pt3.x = pt1.x - pt1.y / k;
	pt3.y = 0;
	pt4.x = 0;
	pt4.y = pt2.y - k * pt2.x;

	for (size_t i = 0; i < w; i++)
	{
		for (size_t j = 0; j < h; j++)
		{
			if (j > (i - pt4.x)*k + pt4.y)
			{
				mask.at<uchar>(j, i) = 255;
			}
		}
	}
	//imshow("mask", mask);
	inputImg.copyTo(outputImg, mask);

}


//------------------------------ransacLines()-----------------------------
//-------------------------------------------------------------------------
void IPSG::CbankDetect::ransacLines(std::vector<cv::Point>& input, std::vector<cv::Vec4d>& lines,
	double distance, unsigned int ngon, unsigned int itmax) {

	if (!input.empty())
		for (int i = 0; i < ngon; ++i) {
			unsigned int Mmax = 0;
			cv::Point2d imax;
			cv::Point2d jmax;
			cv::Vec4d line;
			size_t t1, t2;

			std::random_device rd;     // only used once to initialise (seed) engine
			std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
			std::uniform_int_distribution<int> uni(0, input.size() - 1); // guaranteed unbiased // 概率相同

			unsigned int it = itmax;
			while (--it) {
				t1 = uni(rng);
				t2 = uni(rng);
				t2 = (t1 == t2 ? uni(rng) : t2);
				unsigned int M = 0;
				cv::Point2d i = input[t1];
				cv::Point2d j = input[t2];
				for (auto a : input) {
					double dis = fabs((j.x - i.x)*(a.y - i.y) - (j.y - i.y)*(a.x - i.x)) /
						sqrt((j.x - i.x)*(j.x - i.x) + (j.y - i.y)*(j.y - i.y));

					if (dis < distance)
						++M;
				}
				if (M > Mmax) {
					Mmax = M;
					imax = i;
					jmax = j;
				}
			}
			line[0] = imax.x;
			line[1] = imax.y;
			line[2] = jmax.x;
			line[3] = jmax.y;
			lines.push_back(line);
			auto iter = input.begin();
			while (iter != input.end()) {
				double dis = fabs((jmax.x - imax.x)*((*iter).y - imax.y) -
					(jmax.y - imax.y)*((*iter).x - imax.x))
					/ sqrt((jmax.x - imax.x)*(jmax.x - imax.x)
					+ (jmax.y - imax.y)*(jmax.y - imax.y));
				if (dis < distance)
					iter = input.erase(iter);  //erase the dis within , then point to the next element
				else ++iter;
			}
		}
	else std::cout << "no input to ransacLines" << std::endl;
}



bool IPSG::CbankDetect::threshold_Lab(cv::Mat &inputImg, cv::Mat &outputMask)
{
	//创建阈值调整滑动条
	cv::namedWindow("Control", CV_WINDOW_AUTOSIZE);
	cv::createTrackbar("LowL", "Control", &iLowL, 255);
	//L (0 - 255)
	cv::createTrackbar("HighL", "Control", &iHighL, 255);
	cv::createTrackbar("LowA", "Control", &iLowA, 255);
	//a (0 - 255)
	cv::createTrackbar("HighA", "Control", &iHighA, 255);
	cv::createTrackbar("LowB", "Control", &iLowB, 255);
	//b (0 - 255)
	cv::createTrackbar("HighB", "Control", &iHighB, 255);
	on_Trackbar(0, 0);

	cv::Mat imgLAB, imgThresholded;
	std::vector<cv::Mat> labSplit;
	//颜色空间转换 BGR to Lab
	cv::cvtColor(inputImg, imgLAB, CV_BGR2Lab);
	cv::split(imgLAB, labSplit);
	//equalizeHist(labSplit[0],labSplit[0]);
	//equalizeHist(labSplit[1],labSplit[1]);
	//equalizeHist(labSplit[2],labSplit[2]);
	//imshow("lab_L",labSplit[0]);
	//imshow("lab_a",labSplit[1]);
	//imshow("lab_b",labSplit[2]);
	cv::merge(labSplit, imgLAB);

	//阈值处理
	cv::inRange(imgLAB, cv::Scalar(iLowL, iLowA, iLowB), cv::Scalar(iHighL, iHighA, iHighB), imgThresholded);

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	//开操作 
	cv::morphologyEx(imgThresholded, imgThresholded, cv::MORPH_OPEN, element);
	//闭操作 
	cv::morphologyEx(imgThresholded, imgThresholded, cv::MORPH_CLOSE, element);

	//imshow("Thresholded Image", imgThresholded); 
	std::vector<std::vector<cv::Point> > Contours;
	std::vector<cv::Vec4i> Hierarchy;
	cv::findContours(imgThresholded, Contours, Hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//cout<<"Contour num:"<<Contours.size()<<endl;
	//得到最大轮廓
	int maxContour = 0;
	if (!Contours.empty() && !Hierarchy.empty())
	{
		for (int i = 0; i < Contours.size(); i++)
		{
			if (/*Contours[i].size() > 50 && */Contours[i].size() > Contours[maxContour].size())
			{
				maxContour = i;
			}
		}
		//drawContours(inputImg, Contours, maxContour, Scalar(255, 255, 255), 5);
		cv::Mat mask(cv::Size(inputImg.cols, inputImg.rows), CV_8UC1, cv::Scalar(0));
		
		cv::drawContours(mask, Contours, maxContour, cv::Scalar(255), 1);
		mask.copyTo(outputMask);
		cv::imshow("outputMask", outputMask);
		return 1;
	}
	else
	{
		std::cout << "threshold_Lab failed" << std::endl;
		return 0;
	}
}


	//-----------------threshold_OTSU( )----------------------
	//----------------------------------------------------
	//                
	bool IPSG::CbankDetect::threshold_OTSU(cv::Mat &inputImg, cv::Mat &outputMask)
	{
		cv::Mat imgLab, imgThresholded;
		std::vector<cv::Mat> labSplit;
		//颜色空间转换 BGR to Lab
		cv::cvtColor(inputImg, imgLab, CV_BGR2Lab);
		cv::split(imgLab, labSplit);
		cv::threshold(labSplit[0], labSplit[0], 0, 255, CV_THRESH_OTSU);
		cv::threshold(labSplit[1], labSplit[1], 0, 255, CV_THRESH_OTSU);
		cv::threshold(labSplit[2], labSplit[2], 0, 255, CV_THRESH_OTSU);
		/*cv::imshow("lab_L", labSplit[0]);
		cv::imshow("lab_a", labSplit[1]);
		cv::imshow("lab_b", labSplit[2]);*/
		//Mat mask = ~((labSplit[1]+labSplit[2])& ~labSplit[0]);  //
		//Mat mask = (~labSplit[1] & labSplit[2] + labSplit[0]);//1/27
		//Mat mask = labSplit[1] & ~labSplit[2] ;
		//imgThresholded = ~labSplit[0] + labSplit[1] & ~labSplit[2];//1/21
		//imgThresholded = labSplit[0] & labSplit[1] & ~labSplit[2];//
		cv::Mat invertedImg, cloudImg, bankImg, stoneRailingImg,bridgeImg,treeAndGrassImg;
		//invertedImg = (labSplit[2] & ~labSplit[0]) + (labSplit[2] & ~labSplit[1]);//b&~a+b&~L
		invertedImg = labSplit[2] & ~(labSplit[0] + labSplit[1]);
		stoneRailingImg = labSplit[0] & labSplit[1] & labSplit[2];
		treeAndGrassImg = labSplit[0] & ~labSplit[1] & labSplit[2];
		cloudImg = labSplit[0] & labSplit[1] & ~labSplit[2];
		imgThresholded = ~(invertedImg + cloudImg);
		bankImg = cloudImg& ~invertedImg;
		//cv::imshow("cloud", cloudImg);
		//cv::imshow("invertedImg", invertedImg);
		//cv::imshow("stoneRailingImg", stoneRailingImg);
		//cv::imshow("treeAndGrassImg", treeAndGrassImg);
		//cv::imshow("bankImg", bankImg);

		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		//开操作 
		cv::morphologyEx(imgThresholded, imgThresholded, cv::MORPH_OPEN, element);
		//闭操作 
		cv::morphologyEx(imgThresholded, imgThresholded, cv::MORPH_CLOSE, element);
		cv::imshow("imgThresholded", imgThresholded);

		std::vector<std::vector<cv::Point> > Contours;
		std::vector<cv::Vec4i> Hierarchy;
		cv::findContours(imgThresholded, Contours, Hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		//std::cout<<"Contour num:"<<Contours.size()<<std::endl;
		//得到最大轮廓
		int maxContour = 0;
		if (!Contours.empty() && !Hierarchy.empty())
		{
			for (int i = 0; i < Contours.size(); i++)
			{
				if (/*Contours[i].size() > 50 && */Contours[i].size() > Contours[maxContour].size())
				{
					maxContour = i;
				}
			}
			//cv::drawContours(inputImg, Contours, maxContour, cv::Scalar(255, 255, 255), 5);
			cv::Mat mask(cv::Size(inputImg.cols, inputImg.rows), CV_8UC1, cv::Scalar(0));
			cv::drawContours(mask, Contours, maxContour, cv::Scalar(255),1);
			mask.copyTo(outputMask);
			cv::imshow("outputMask", outputMask);
			return 1;
		}
		else
		{
			std::cout << "threshold_Lab failed" << std::endl;
			return 0;
		}
	}




//-----------------bankDetect( )----------------------
//----------------------------------------------------
//                岸体检测与分割
bool IPSG::CbankDetect::bankDetect(cv::Mat &inputImg, cv::Mat &outputImg, int thresholdMethod)
{
	cv::Mat mask(cv::Size(inputImg.cols, inputImg.rows), CV_8UC1, cv::Scalar(0));
	std::vector<cv::Point> inputPoint;
	static int thresholdFlag;
	if (0==thresholdMethod)
	{
		std::cout << "using Lab" << std::endl;
		if (threshold_Lab(inputImg, mask))
			thresholdFlag = 1;
		else
			thresholdFlag = 0;
	}
	else if (1==thresholdMethod)
	{
		std::cout << "using OTSU" << std::endl;
		if (threshold_OTSU(inputImg, mask))
			thresholdFlag = 1;
		else
			thresholdFlag = 0;
	}
	
	if (thresholdFlag)
	{
		//得到分界线离散点集
		getPoint(mask, inputPoint);
		////cout << "inputPoint" << inputPoint << endl;
		////cout << "Contours" << Contours[maxContour] << endl;
		//画出离散点
		for (size_t i = 0; i < inputPoint.size(); i++)
		{
			cv::circle(mask, inputPoint[i], 3, cv::Scalar(255), 3);
		}
		cv::imshow("point_on_mask", mask);
		//RANSAC直线拟合
		std::vector<cv::Vec4d> Lines;
		if (inputPoint.size() > 80)//大于80个点才做RANSAC
		{
			ransacLines(inputPoint, Lines, 10, 1, 100);//拟合一条直线，迭代100次

			for (size_t i = 0; i < Lines.size(); i++)
			{
				//cout<<"Lines"<<i<<":"<<Lines[i]<<endl;
				//画出分界线
				//drawExtendLine(inputImg, Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]),p1,p2, Scalar(0, 255, 255), 2, 8);
				//图像分割
				segment(inputImg, cv::Point(Lines[i][0], Lines[i][1]), cv::Point(Lines[i][2], Lines[i][3]), outputImg);
			}
			//std::cout << "success" << std::endl;
			return 1;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		return 0;
	}
}





