#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

// IMPLEMENTATION OF LINEAR MONGE-KANTOROVITCH IMAGE
// COLOUR TRANSFER PROCESSING.

// Copyright © Terry Johnson June 2020
// https://github.com/TJCoding


// References:
// Pitie F and A. Kokaram. "The linear Mange-Kantovitch linear
// colour mapping for example-based colour transfer."
// In 4th IEE European Conference on Visual Media Production (CVMP07)
// London November 2007.

// Runs under OpenCV 2.4.13.7


cv::Mat LMK(cv::Mat imgs, cv::Mat imgt);


int main()
{

// ##########################################################################
// #######################  PROCESSING SELECTIONS  ##########################
// ##########################################################################

    // Specify the image files that are to be processed,
    // where 'source image' provides the colour scheme that
    // is to be applied to 'target image'.

    std::string targetname = "images/Flowers_target.jpg";
    std::string sourcename = "images/Flowers_source.jpg";

// ###########################################################################
// ###########################################################################
// ###########################################################################

    // Read in the files.
    cv::Mat target = cv::imread(targetname,1);
    cv::Mat source = cv::imread(sourcename,1);

    target=LMK(source,target);

    // Display and save the final image.
    cv::imshow("processed image",target);
    cv::imwrite("images/processed.jpg",target);
    cv::waitKey(0);

	return 0;
}

cv::Mat LMK(cv::Mat imgs,cv::Mat imgt)
{
    cv::Mat cov_t, cov_s, means_t, means_s;
    cv::Mat Da2, Ua, Dc2, Uc, C, T;
    cv::Mat Da=cv::Mat::zeros(3,3,CV_64FC1);
    cv::Mat Dc=cv::Mat::zeros(3,3,CV_64FC1);
    cv::Mat Da_inv=cv::Mat::zeros(3,3,CV_64FC1);
    cv::Mat result;

    // Convert the images to floating point and then
    // compute the respective cross covariance matrices.
	imgs.convertTo(imgs, CV_32FC3, 1/255.0);
	imgt.convertTo(imgt, CV_32FC3, 1/255.0);

    cv::calcCovarMatrix(imgt.reshape(1, imgt.cols * imgt.rows),
                  cov_t, means_t, CV_COVAR_ROWS | CV_COVAR_NORMAL);
    cv::calcCovarMatrix(imgs.reshape(1, imgs.cols * imgs.rows),
                  cov_s, means_s, CV_COVAR_ROWS | CV_COVAR_NORMAL);

    // The following is an implementation of Pitie's 'MLK' function
    // using the same notation.
    // https://github.com/frcs/colour-transfer
    //
    // The eigenvector matrix as computed in
    // OpenCV is the transpose of that in Matlab.
    //
    cv::eigen(cov_t, Da2, Ua);
    cv::sqrt(max(Da2,10^-20),Da2);
    Da=(cv::Mat_<double>(3,3)
                 <<  Da2.at<double>(0, 0),0,0,
                     0,Da2.at<double>(1, 0),0,
                     0,0,Da2.at<double>(2, 0));
    C=Da*Ua*cov_s*Ua.t()*Da;
    cv::eigen(C, Dc2, Uc);
    cv::sqrt(max(Dc2,10^-20),Dc2);
    Dc=(cv::Mat_<double>(3,3)
                 <<  Dc2.at<double>(0, 0),0,0,
                     0,Dc2.at<double>(1, 0),0,
                     0,0,Dc2.at<double>(2, 0));
    Da_inv=(cv::Mat_<double>(3,3)
                 <<  1.0/Da2.at<double>(0, 0),0,0,
                     0,1.0/Da2.at<double>(1, 0),0,
                     0,0,1.0/Da2.at<double>(2, 0));
    T=Ua.t()*Da_inv*Uc.t()*Dc*Uc*Da_inv*Ua;


    transform(imgt-means_t, result,T);
    result=result + means_s;

    result.convertTo(result, CV_8UC3,255.0);

	return result;
}

