/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{



/**
 * @brief Sim3Solver::Sim3Solver
 * @param pKF1 当前帧
 * @param pKF2 闭环帧
 * @param vpMatched12 匹配的地图点,索引和当前帧角点的索引对应
 * @param bFixScale 对于单目,尺度可以缩放,为false
 */
Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale):
    mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    //两帧
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    vector<MapPoint*> vpKeyFrameMP1 = pKF1->GetMapPointMatches();//当前帧的地图点

    mN1 = vpMatched12.size();//当前帧角点个数

    mvpMapPoints1.reserve(mN1);//当前帧 匹配好的地图点
    mvpMapPoints2.reserve(mN1);//闭环帧 匹配好的地图点
    mvpMatches12 = vpMatched12;//匹配点,索引为当前帧角点索引对应,值为帧2的地图点
    mvnIndices1.reserve(mN1);//匹配点对应于mvpMatches12中的索引,索引和mvpMapPoints1等对应,值为mvpMatches12的索引,即当前帧角点索引
    mvX3Dc1.reserve(mN1);//当前帧地图点,在当前帧坐标系内的位置
    mvX3Dc2.reserve(mN1);//闭环帧地图点,在闭环帧坐标系内的位置

    cv::Mat Rcw1 = pKF1->GetRotation();//当前帧的旋转矩阵
    cv::Mat tcw1 = pKF1->GetTranslation();//当前帧 平移
    cv::Mat Rcw2 = pKF2->GetRotation();//闭环帧, 旋转
    cv::Mat tcw2 = pKF2->GetTranslation();//闭环帧,平移

    mvAllIndices.reserve(mN1);

    size_t idx=0;
    for(int i1=0; i1<mN1; i1++)//遍历匹配点
    {
        if(vpMatched12[i1])
        {
            MapPoint* pMP1 = vpKeyFrameMP1[i1];//帧1的地图点
            MapPoint* pMP2 = vpMatched12[i1];//通过角点匹配，帧2和帧1匹配的地图点,理想下应该相等,由于两帧没有共视,故是误差来源

            if(!pMP1)
                continue;

            if(pMP1->isBad() || pMP2->isBad())
                continue;

            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);//地图点在当前帧的角点索引
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);//地图点在闭环帧的角点索引

            if(indexKF1<0 || indexKF2<0)
                continue;

            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];//当前帧角点
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];//闭环帧角点

            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];//图像金字塔相关，缩放因子的平方
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            //mvpMapPoints1，mvpMapPoints2，mvnMaxError1，mvnMaxError2，mvX3Dc1，mvX3Dc2应该是对应关系
            mvnMaxError1.push_back(9.210*sigmaSquare1);
            mvnMaxError2.push_back(9.210*sigmaSquare2);

            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);

            cv::Mat X3D1w = pMP1->GetWorldPos();
            mvX3Dc1.push_back(Rcw1*X3D1w+tcw1);//地图点1在帧1相机坐标系内的坐标

            cv::Mat X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Rcw2*X3D2w+tcw2);//地图点2在帧2相机坐标系内的坐标

            mvAllIndices.push_back(idx);//对应于角点的位置
            idx++;
        }
    }

    mK1 = pKF1->mK;//相机内参，用于根据3d坐标，计算投影位置
    mK2 = pKF2->mK;

    /**
     * @brief FromCameraToImage
     * mvX3Dc1--相对于相机坐标系的位置
     * mvP1im1--投影位置
     * mK1--相机内参
     */
    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);//3d点在帧上投影的位置
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);

    SetRansacParameters();
}

/**
 * @brief Sim3Solver::SetRansacParameters 可以参考 http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf
 * @param probability 选择正确的概率，默认0.99，
 * @param minInliers 最少内点的个数，除以总数就是内点的概率. 默认6
 * @param maxIterations 最大迭代次数，默认300，实际上，迭代次数需要计算，但计算结果如果大于该值，就按该值来算
 */
void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;    

    N = mvpMapPoints1.size(); // number of correspondences 关联地图点的个数

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mnIterations = 0;
}


/**
 * @brief Sim3Solver::iterate
 * @param nIterations 用户指定的本次最大迭代次数，如果它小于最大迭代次数，也可能导致bNoMore为false,但返回了空值
 * @param bNoMore 输出数据，为true，表示内点数不符合期望，迭代失败。返回false，说明成功，或者还可以继续迭代
 * @param vbInliers
 * @param nInliers
 * @return
 */
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1,false);//索引和帧1角点索引相同,表示该帧是是否是通过sim3位姿估计后的内点
    nInliers=0;

    if(N<mRansacMinInliers)//mRansacMinInliers为指定的内点个数
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    cv::Mat P3Dc1i(3,3,CV_32F);//当前帧中,地图点的相对坐标,每列对应一个点
    cv::Mat P3Dc2i(3,3,CV_32F);//候选帧中,地图点的相对坐标,每列对应一个点

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;//本次函数调用,迭代次数
        mnIterations++;//累积迭代次数

        vAvailableIndices = mvAllIndices;//索引和值相同,用来产生随机数

        // Get min set of points
        //得到3对随机点
        for(short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        ComputeSim3(P3Dc1i,P3Dc2i);//根据这三对点,估计相对位姿

        CheckInliers();//根据相对位姿，计算投影误差，检查内点个数

        if(mnInliersi>=mnBestInliers)
        {
            //记录最多内点对应的情况
            mvbBestInliers = mvbInliersi;//记录是否是内点
            mnBestInliers = mnInliersi;//最多内点个数
            mBestT12 = mT12i.clone();//相对位姿，从2变到1
            mBestRotation = mR12i.clone();//x旋转
            mBestTranslation = mt12i.clone();//平移
            mBestScale = ms12i;//缩放

            if(mnInliersi>mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)
                    if(mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;
                return mBestT12;
            }
        }
    }

    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;

    return cv::Mat();
}

cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers12,nInliers);
}

void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    cv::reduce(P,C,1,CV_REDUCE_SUM);
    C = C/P.cols;

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;
    }
}

void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    //参考英文注释里的文章:
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1 三点中心
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    ComputeCentroid(P1,Pr1,O1);//O1-三点中心,Pr1--三点相对于中心O1的位置,
    ComputeCentroid(P2,Pr2,O2);//同上

    // Step 2: Compute M matrix

    cv::Mat M = Pr2*Pr1.t();//矩阵M

    // Step 3: Compute N matrix
    //计算矩阵N

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;//eval--特征值,按从大到小排列,evec--对应的特征向量

    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation

    cv::Mat vec(1,3,evec.type());//最终结果为表示旋转的向量,向量方向为旋转轴,向量大小为绕该轴转过的角度
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    double ang=atan2(norm(vec),evec.at<float>(0,0));//旋转角

    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half 旋转角度表示,方向为旋转轴,大小为绕该轴的角度

    mR12i.create(3,3,P1.type());//旋转矩阵

    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis 从2变到1的旋转矩阵，注意还没算平移哦

    // Step 5: Rotate set 2

    cv::Mat P3 = mR12i*Pr2;

    // Step 6: Scale

    if(!mbFixScale)
    {
        //单目，要执行这里
        double nom = Pr1.dot(P3);//看成9维矢量点乘，含s的一次方。疑问：平移分量为啥没算进去？
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        cv::pow(P3,2,aux_P3);
        double den = 0;

        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);
            }
        }

        ms12i = nom/den;// 缩放因子
        //std::cout<<"debug 缩放因子："<<ms12i<<std::endl;

    }
    else
        ms12i = 1.0f;

    // Step 7: Translation

    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2;

    // Step 8: Transformation

    // Step 8.1 T12
    mT12i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sR = ms12i*mR12i;

    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));

    // Step 8.2 T21

    mT21i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));
}

/**
 * @brief Sim3Solver::CheckInliers  判断哪些地图点对是内点
 * 误差衡量标准:
 * 利用相对位姿,把地图点2在帧2的位置,变换到帧1,然后,投影到帧1上(vP2im1),再和地图点1的投影(mvP1im1),得到帧1的投影误差
 * 同样的方法,得到帧2的投影误差
 * 两个误差都满足一定阈值的时候,判断为内点
 * 阈值的大小,和图像金子塔有关,或者,和角点的大小有关
 */
void Sim3Solver::CheckInliers()
{
    /*将闭环帧的地图点投影到当前帧上,将当前这的地图点投影到闭环帧上
     * mvX3Dc2,mvX3Dc1--地图点的相对坐标
     * mT12i,mT21i--两帧相对位姿
     * mK1,mK2--相机内参
     * */
    vector<cv::Mat> vP1im2, vP2im1;
    Project(mvX3Dc2,vP2im1,mT12i,mK1);//将点2投影到相机1的帧上 
    Project(mvX3Dc1,vP1im2,mT21i,mK2);//将点1投影到相机2的帧上

    mnInliersi=0;

    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}


cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}


/**
 * @brief Sim3Solver::Project
 * @param vP3Dw 3d点的相对坐标
 * @param vP2D vP3Dw在相机上的投影
 * @param Tcw 位姿，从3d点坐标所在坐标系，通过该矩阵变换，变为当前相机坐标系
 * @param K 当前相机坐标系的相机内参
 */
void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;
        const float invz = 1/(P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0)*invz;
        const float y = P3Dc.at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

/**
 * @brief Sim3Solver::FromCameraToImage 根据3d坐标，求出在图片上的投影位置，这个3d坐标的参照系为相机坐标系
 * @param vP3Dc 相对坐标
 * @param vP2D 输出,投影坐标
 * @param K 相机坐标
 */
void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
    {
        const float invz = 1/(vP3Dc[i].at<float>(2));
        const float x = vP3Dc[i].at<float>(0)*invz;
        const float y = vP3Dc[i].at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

} //namespace ORB_SLAM
