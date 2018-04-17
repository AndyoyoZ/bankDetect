//#include "Scole.h"
#include "bankDetect.h"
//#include <io.h>
int main()
{	
	char img_name[20];
	int count = 0;
	do
	{

		std::sprintf(img_name, "%s%04d%s", "../data/image", count++, ".jpg");

		cv::Mat srcImage = cv::imread(img_name, 1);
		if (!srcImage.data)
		{
			std::cout << "Image Load error!" << std::endl;
			//return 0;
			break;
		}
		cv::Mat resutImage;
		IPSG::CbankDetect cbankdetect;
		if (cbankdetect.bankDetect(srcImage, resutImage, 1))
		{
			cv::imshow("resutImage", resutImage);
			std::cout << "success" << std::endl;
		}
		else
			std::cout << "failed" << std::endl;

		cv::imshow("src", srcImage);
		//cv::waitKey(500);
	} while (cv::waitKey() == 32);//space
	
	//cv::waitKey(0);
	cv::destroyAllWindows();
	//system("pause");
	return 0;
}
