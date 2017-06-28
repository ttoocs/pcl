/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include <pcl/gpu/kinfu_large_scale/kinfu.h>
#include "estimate_combined.h"
#include "internal.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#ifdef HAVE_OPENCV
  #include <opencv2/opencv.hpp>
  #include <opencv2/core/eigen.hpp>
  //#include <opencv2/gpu/gpu.hpp>
  using namespace cv;
#endif

#include <limits>

using namespace std;
using namespace pcl::device::kinfuLS;

//////////////////////////////
// RGBDOdometry part
//////////////////////////////

inline static
	void computeC_RigidBodyMotion( double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy )
{
	double invz  = 1. / p3d.z,
		v0 = dIdx * fx * invz,
		v1 = dIdy * fy * invz,
		v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

	C[0] = -p3d.z * v1 + p3d.y * v2;
	C[1] =  p3d.z * v0 - p3d.x * v2;
	C[2] = -p3d.y * v0 + p3d.x * v1;
	C[3] = v0;
	C[4] = v1;
	C[5] = v2;
}

inline static
	void computeProjectiveMatrix( const Mat& ksi, Mat& Rt )
{
	CV_Assert( ksi.size() == Size(1,6) && ksi.type() == CV_64FC1 );

	// for infinitesimal transformation
	Rt = Mat::eye(4, 4, CV_64FC1);

	Mat R = Rt(Rect(0,0,3,3));
	Mat rvec = ksi.rowRange(0,3);

	Rodrigues( rvec, R );

	Rt.at<double>(0,3) = ksi.at<double>(3);
	Rt.at<double>(1,3) = ksi.at<double>(4);
	Rt.at<double>(2,3) = ksi.at<double>(5);
}

static
	void cvtDepth2Cloud( const Mat& depth, Mat& cloud, const Mat& cameraMatrix )
{
	//CV_Assert( cameraMatrix.type() == CV_64FC1 );
	const double inv_fx = 1.f/cameraMatrix.at<double>(0,0);
	const double inv_fy = 1.f/cameraMatrix.at<double>(1,1);
	const double ox = cameraMatrix.at<double>(0,2);
	const double oy = cameraMatrix.at<double>(1,2);
	cloud.create( depth.size(), CV_32FC3 );
	for( int y = 0; y < cloud.rows; y++ )
	{
		Point3f* cloud_ptr = reinterpret_cast<Point3f*>(cloud.ptr(y));
		const float* depth_prt = reinterpret_cast<const float*>(depth.ptr(y));
		for( int x = 0; x < cloud.cols; x++ )
		{
			float z = depth_prt[x];
			cloud_ptr[x].x = (float)((x - ox) * z * inv_fx);
			cloud_ptr[x].y = (float)((y - oy) * z * inv_fy);
			cloud_ptr[x].z = z;
		}
	}
}

static inline
	void set2shorts( int& dst, int short_v1, int short_v2 )
{
	unsigned short* ptr = reinterpret_cast<unsigned short*>(&dst);
	ptr[0] = static_cast<unsigned short>(short_v1);
	ptr[1] = static_cast<unsigned short>(short_v2);
}

static inline
	void get2shorts( int src, int& short_v1, int& short_v2 )
{
	typedef union { int vint32; unsigned short vuint16[2]; } s32tou16;
	const unsigned short* ptr = (reinterpret_cast<s32tou16*>(&src))->vuint16;
	short_v1 = ptr[0];
	short_v2 = ptr[1];
}

static
	int computeCorresp( const Mat& K, const Mat& K_inv, const Mat& Rt,
	const Mat& depth0, const Mat& depth1, const Mat& texturedMask1, float maxDepthDiff,
	Mat& corresps )
{
	CV_Assert( K.type() == CV_64FC1 );
	CV_Assert( K_inv.type() == CV_64FC1 );
	CV_Assert( Rt.type() == CV_64FC1 );

	corresps.create( depth1.size(), CV_32SC1 );

	Mat R = Rt(Rect(0,0,3,3)).clone();

	Mat KRK_inv = K * R * K_inv;
	const double * KRK_inv_ptr = reinterpret_cast<const double *>(KRK_inv.ptr());

	Mat Kt = Rt(Rect(3,0,1,3)).clone();
	Kt = K * Kt;
	const double * Kt_ptr = reinterpret_cast<const double *>(Kt.ptr());

	Rect r(0, 0, depth1.cols, depth1.rows);

	corresps = Scalar(-1);
	int correspCount = 0;
	for( int v1 = 0; v1 < depth1.rows; v1++ )
	{
		for( int u1 = 0; u1 < depth1.cols; u1++ )
		{
			float d1 = depth1.at<float>(v1,u1);
			if( !cvIsNaN(d1) && texturedMask1.at<uchar>(v1,u1) )
			{
				float transformed_d1 = (float)(d1 * (KRK_inv_ptr[6] * u1 + KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8]) + Kt_ptr[2]);
				int u0 = cvRound((d1 * (KRK_inv_ptr[0] * u1 + KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2]) + Kt_ptr[0]) / transformed_d1);
				int v0 = cvRound((d1 * (KRK_inv_ptr[3] * u1 + KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5]) + Kt_ptr[1]) / transformed_d1);

				if( r.contains(Point(u0,v0)) )
				{
					float d0 = depth0.at<float>(v0,u0);
					if( !cvIsNaN(d0) && std::abs(transformed_d1 - d0) <= maxDepthDiff )
					{
						int c = corresps.at<int>(v0,u0);
						if( c != -1 )
						{
							int exist_u1, exist_v1;
							get2shorts( c, exist_u1, exist_v1);

							float exist_d1 = (float)(depth1.at<float>(exist_v1,exist_u1) * (KRK_inv_ptr[6] * exist_u1 + KRK_inv_ptr[7] * exist_v1 + KRK_inv_ptr[8]) + Kt_ptr[2]);

							if( transformed_d1 > exist_d1 )
								continue;
						}
						else
							correspCount++;

						set2shorts( corresps.at<int>(v0,u0), u1, v1 );
					}
				}
			}
		}
	}

	return correspCount;
}

static inline
	void preprocessDepth( Mat depth0, Mat depth1,
	const Mat& validMask0, const Mat& validMask1,
	float minDepth, float maxDepth )
{
	CV_DbgAssert( depth0.size() == depth1.size() );

	for( int y = 0; y < depth0.rows; y++ )
	{
		for( int x = 0; x < depth0.cols; x++ )
		{
			float& d0 = depth0.at<float>(y,x);
			if( !cvIsNaN(d0) && (d0 > maxDepth || d0 < minDepth || d0 <= 0 || (!validMask0.empty() && !validMask0.at<uchar>(y,x))) )
				d0 = std::numeric_limits<float>::quiet_NaN();

			float& d1 = depth1.at<float>(y,x);
			if( !cvIsNaN(d1) && (d1 > maxDepth || d1 < minDepth || d1 <= 0 || (!validMask1.empty() && !validMask1.at<uchar>(y,x))) )
				d1 = std::numeric_limits<float>::quiet_NaN();
		}
	}
}

static
	void buildPyramids( const Mat& image0, const Mat& image1,
	const Mat& depth0, const Mat& depth1,
	const Mat& cameraMatrix, int sobelSize, double sobelScale,
	const vector<float>& minGradMagnitudes,
	vector<Mat>& pyramidImage0, vector<Mat>& pyramidDepth0,
	vector<Mat>& pyramidImage1, vector<Mat>& pyramidDepth1,
	vector<Mat>& pyramid_dI_dx1, vector<Mat>& pyramid_dI_dy1,
	vector<Mat>& pyramidTexturedMask1, vector<Mat>& pyramidCameraMatrix )
{
	const int pyramidMaxLevel = (int)minGradMagnitudes.size() - 1;

	buildPyramid( image0, pyramidImage0, pyramidMaxLevel );
	buildPyramid( image1, pyramidImage1, pyramidMaxLevel );

	pyramid_dI_dx1.resize( pyramidImage1.size() );
	pyramid_dI_dy1.resize( pyramidImage1.size() );
	pyramidTexturedMask1.resize( pyramidImage1.size() );

	pyramidCameraMatrix.reserve( pyramidImage1.size() );

	Mat cameraMatrix_dbl;
	cameraMatrix.convertTo( cameraMatrix_dbl, CV_64FC1 );

	for( size_t i = 0; i < pyramidImage1.size(); i++ )
	{
		Sobel( pyramidImage1[i], pyramid_dI_dx1[i], CV_16S, 1, 0, sobelSize );
		Sobel( pyramidImage1[i], pyramid_dI_dy1[i], CV_16S, 0, 1, sobelSize );

		const Mat& dx = pyramid_dI_dx1[i];
		const Mat& dy = pyramid_dI_dy1[i];

		Mat texturedMask( dx.size(), CV_8UC1, Scalar(0) );
		const float minScalesGradMagnitude2 = (float)((minGradMagnitudes[i] * minGradMagnitudes[i]) / (sobelScale * sobelScale));
		for( int y = 0; y < dx.rows; y++ )
		{
			for( int x = 0; x < dx.cols; x++ )
			{
				float m2 = (float)(dx.at<short>(y,x)*dx.at<short>(y,x) + dy.at<short>(y,x)*dy.at<short>(y,x));
				if( m2 >= minScalesGradMagnitude2 )
					texturedMask.at<uchar>(y,x) = 255;
			}
		}
		pyramidTexturedMask1[i] = texturedMask;
		Mat levelCameraMatrix = i == 0 ? cameraMatrix_dbl : 0.5f * pyramidCameraMatrix[i-1];
		levelCameraMatrix.at<double>(2,2) = 1.;
		pyramidCameraMatrix.push_back( levelCameraMatrix );
	}

	buildPyramid( depth0, pyramidDepth0, pyramidMaxLevel );
	buildPyramid( depth1, pyramidDepth1, pyramidMaxLevel );
}

static
	bool solveSystem( const Mat& C, const Mat& dI_dt, double detThreshold, Mat& ksi, Eigen::Matrix<float, 6, 6, Eigen::RowMajor> & AA, Eigen::Matrix<float, 6, 1> & bb )
{
	Mat A = C.t() * C;
	Mat B = -C.t() * dI_dt;

	cv2eigen( A, AA );
	cv2eigen( B, bb );

	double det = cv::determinant(A);

	if( fabs (det) < detThreshold || cvIsNaN(det) || cvIsInf(det) )
		return false;

	cv::solve( A, B, ksi, DECOMP_CHOLESKY );
	return true;
}

typedef void (*ComputeCFuncPtr)( double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy );

static
	bool computeKsi( int transformType,
	const Mat& image0, const Mat&  cloud0,
	const Mat& image1, const Mat& dI_dx1, const Mat& dI_dy1,
	const Mat& corresps, int correspsCount,
	double fx, double fy, double sobelScale, double determinantThreshold,
	Mat& ksi,
	Eigen::Matrix<float, 6, 6, Eigen::RowMajor> & AA, Eigen::Matrix<float, 6, 1> & bb )
{
	int Cwidth = -1;
	ComputeCFuncPtr computeCFuncPtr = 0;
	computeCFuncPtr = computeC_RigidBodyMotion;
	Cwidth = 6;
	Mat C( correspsCount, Cwidth, CV_64FC1 );
	Mat dI_dt( correspsCount, 1, CV_64FC1 );

	double sigma = 0;
	int pointCount = 0;
	for( int v0 = 0; v0 < corresps.rows; v0++ )
	{
		for( int u0 = 0; u0 < corresps.cols; u0++ )
		{
			if( corresps.at<int>(v0,u0) != -1 )
			{
				int u1, v1;
				get2shorts( corresps.at<int>(v0,u0), u1, v1 );
				double diff = static_cast<double>(image1.at<uchar>(v1,u1)) -
					static_cast<double>(image0.at<uchar>(v0,u0));
				sigma += diff * diff;
				pointCount++;
			}
		}
	}
	sigma = std::sqrt(sigma/pointCount);

	pointCount = 0;
	for( int v0 = 0; v0 < corresps.rows; v0++ )
	{
		for( int u0 = 0; u0 < corresps.cols; u0++ )
		{
			if( corresps.at<int>(v0,u0) != -1 )
			{
				int u1, v1;
				get2shorts( corresps.at<int>(v0,u0), u1, v1 );

				double diff = static_cast<double>(image1.at<uchar>(v1,u1)) -
					static_cast<double>(image0.at<uchar>(v0,u0));
				double w = sigma + std::abs(diff);
				w = w > DBL_EPSILON ? 1./w : 1.;

				(*computeCFuncPtr)( (double*)C.ptr(pointCount),
					w * sobelScale * dI_dx1.at<short int>(v1,u1),
					w * sobelScale * dI_dy1.at<short int>(v1,u1),
					cloud0.at<Point3f>(v0,u0), fx, fy);

				dI_dt.at<double>(pointCount) = w * diff;
				pointCount++;
			}
		}
	}

	Mat sln;
	bool solutionExist = solveSystem( C, dI_dt, determinantThreshold, sln, AA, bb );

	/*
	cout << C << endl;
	cout << dI_dt << endl;
	cout << sln << endl;

	cout << endl;
	*/

	if( solutionExist )
	{
		ksi.create(6,1,CV_64FC1);
		ksi = Scalar(0);

		Mat subksi;
		subksi = ksi;
		sln.copyTo( subksi );
	}

	return solutionExist;
}

//////////////////////////////
// RGBDOdometry part end
//////////////////////////////


using namespace std;
using namespace pcl::device::kinfuLS;

using pcl::device::kinfuLS::device_cast;
using Eigen::AngleAxisf;
using Eigen::Array3f;
using Eigen::Vector3i;
using Eigen::Vector3f;

namespace pcl
{
  namespace gpu
  {
    namespace kinfuLS
    {
      Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);                 
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::kinfuLS::KinfuTracker::KinfuTracker (const Eigen::Vector3f &volume_size, const float shiftingDistance, int rows, int cols) : 
  cyclical_( DISTANCE_THRESHOLD, VOLUME_SIZE, VOLUME_X), rows_(rows), cols_(cols), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), perform_last_scan_ (false), finished_(false), lost_ (false), disable_icp_ (false)
{
  //const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
  const Vector3i volume_resolution (VOLUME_X, VOLUME_Y, VOLUME_Z);

  volume_size_ = volume_size(0);

  tsdf_volume_ = TsdfVolume::Ptr ( new TsdfVolume(volume_resolution) );
  tsdf_volume_->setSize (volume_size);
  
  shifting_distance_ = shiftingDistance;

  // set cyclical buffer values
  cyclical_.setDistanceThreshold (shifting_distance_);
  cyclical_.setVolumeSize (volume_size_, volume_size_, volume_size_);

  setDepthIntrinsics (FOCAL_LENGTH, FOCAL_LENGTH); // default values, can be overwritten
  
  init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

  const int iters[] = {10, 5, 4};
  std::copy (iters, iters + LEVELS, icp_iterations_);

  const float default_distThres = 0.10f; //meters
  const float default_angleThres = sin (20.f * 3.14159254f / 180.f);
  const float default_tranc_dist = 0.03f; //meters

  setIcpCorespFilteringParams (default_distThres, default_angleThres);
  tsdf_volume_->setTsdfTruncDist (default_tranc_dist);

  allocateBufffers (rows, cols);

  rmats_.reserve (30000);
  tvecs_.reserve (30000);

  reset ();
  
  // initialize cyclical buffer
  cyclical_.initBuffer(tsdf_volume_);
  
  last_estimated_rotation_= Eigen::Matrix3f::Identity ();
  last_estimated_translation_= volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setDepthIntrinsics (float fx, float fy, float cx, float cy)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = (cx == -1) ? cols_/2-0.5f : cx;
  cy_ = (cy == -1) ? rows_/2-0.5f : cy;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setInitialCameraPose (const Eigen::Affine3f& pose)
{
  init_Rcam_ = pose.rotation ();
  init_tcam_ = pose.translation ();
  //reset (); // (already called in constructor)
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setDepthTruncationForICP (float max_icp_distance)
{
  max_icp_distance_ = max_icp_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setCameraMovementThreshold(float threshold)
{
  integration_metric_threshold_ = threshold;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::setIcpCorespFilteringParams (float distThreshold, float sineOfAngle)
{
  distThres_  = distThreshold; //mm
  angleThres_ = sineOfAngle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::kinfuLS::KinfuTracker::cols ()
{
  return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::kinfuLS::KinfuTracker::rows ()
{
  return (rows_);
}

void
pcl::gpu::kinfuLS::KinfuTracker::extractAndSaveWorld ()
{
  
  //extract current volume to world model
  PCL_INFO("Extracting current volume...");
  cyclical_.checkForShift(tsdf_volume_, getCameraPose (), 0.6 * volume_size_, true, true, true); // this will force the extraction of the whole cube.
  PCL_INFO("Done\n");
  
  finished_ = true; // TODO maybe we could add a bool param to prevent kinfuLS from exiting after we saved the current world model
  
  int cloud_size = 0;
  
  cloud_size = cyclical_.getWorldModel ()->getWorld ()->points.size();
  
  if (cloud_size <= 0)
  {
    PCL_WARN ("World model currently has no points. Skipping save procedure.\n");
    return;
  }
  else
  {
    PCL_INFO ("Saving current world to world.pcd with %d points.\n", cloud_size);
    pcl::io::savePCDFile<pcl::PointXYZI> ("world.pcd", *(cyclical_.getWorldModel ()->getWorld ()), true);
    return;
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::reset ()
{
  cout << "in reset function!" << std::endl;
  
  if (global_time_)
    PCL_WARN ("Reset\n");
    
  // dump current world to a pcd file
  /*
  if (global_time_)
  {
    PCL_INFO ("Saving current world to current_world.pcd\n");
    pcl::io::savePCDFile<pcl::PointXYZI> ("current_world.pcd", *(cyclical_.getWorldModel ()->getWorld ()), true);
    // clear world model
    cyclical_.getWorldModel ()->reset ();
  }
  */
  
  // clear world model
  cyclical_.getWorldModel ()->reset ();
  
  
  global_time_ = 0;
  rmats_.clear ();
  tvecs_.clear ();

  rmats_.push_back (init_Rcam_);
  tvecs_.push_back (init_tcam_);

  tsdf_volume_->reset ();
  
  // reset cyclical buffer as well
  cyclical_.resetBuffer (tsdf_volume_);
  
  if (color_volume_) // color integration mode is enabled
    color_volume_->reset ();    
  
  // reset estimated pose
  last_estimated_rotation_= Eigen::Matrix3f::Identity ();
  last_estimated_translation_= Vector3f (volume_size_, volume_size_, volume_size_) * 0.5f - Vector3f (0, 0, volume_size_ / 2 * 1.2f);
  
  
  lost_=false;
  has_shifted_=false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::allocateBufffers (int rows, int cols)
{    
  depths_curr_.resize (LEVELS);
  vmaps_g_curr_.resize (LEVELS);
  nmaps_g_curr_.resize (LEVELS);

  vmaps_g_prev_.resize (LEVELS);
  nmaps_g_prev_.resize (LEVELS);

  vmaps_curr_.resize (LEVELS);
  nmaps_curr_.resize (LEVELS);

  vmaps_prev_.resize (LEVELS);
  nmaps_prev_.resize (LEVELS);
  
  coresps_.resize (LEVELS);

  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    depths_curr_[i].create (pyr_rows, pyr_cols);

    vmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);

    vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

    vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_curr_[i].create (pyr_rows*3, pyr_cols);

    vmaps_prev_[i].create (pyr_rows*3, pyr_cols);
    nmaps_prev_[i].create (pyr_rows*3, pyr_cols);
    
    coresps_[i].create (pyr_rows, pyr_cols);
  }  
  depthRawScaled_.create (rows, cols);
  // see estimate tranform for the magic numbers
  int r = (int)ceil ( ((float)rows) / ESTIMATE_COMBINED_CUDA_GRID_Y );
  int c = (int)ceil ( ((float)cols) / ESTIMATE_COMBINED_CUDA_GRID_X );
  gbuf_.create (27, r * c);
  sumbuf_.create (27);
}

inline void 
pcl::gpu::kinfuLS::KinfuTracker::convertTransforms (Matrix3frm& rotation_in_1, Matrix3frm& rotation_in_2, Vector3f& translation_in_1, Vector3f& translation_in_2, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out_1, float3& translation_out_2)
{
  rotation_out_1 = device_cast<Mat33> (rotation_in_1);
  rotation_out_2 = device_cast<Mat33> (rotation_in_2);
  translation_out_1 = device_cast<float3>(translation_in_1);
  translation_out_2 = device_cast<float3>(translation_in_2);
}

inline void 
pcl::gpu::kinfuLS::KinfuTracker::convertTransforms (Matrix3frm& rotation_in_1, Matrix3frm& rotation_in_2, Vector3f& translation_in, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out)
{
  rotation_out_1 = device_cast<Mat33> (rotation_in_1);
  rotation_out_2 = device_cast<Mat33> (rotation_in_2);
  translation_out = device_cast<float3>(translation_in);
}

inline void 
pcl::gpu::kinfuLS::KinfuTracker::convertTransforms (Matrix3frm& rotation_in, Vector3f& translation_in, Mat33& rotation_out, float3& translation_out)
{
  rotation_out = device_cast<Mat33> (rotation_in);  
  translation_out = device_cast<float3>(translation_in);
}

inline void
pcl::gpu::kinfuLS::KinfuTracker::prepareMaps (const DepthMap& depth_raw, const Intr& cam_intrinsics)
{
  // blur raw map
  bilateralFilter (depth_raw, depths_curr_[0]);

  // optionally remove points that are farther than a threshold
  if (max_icp_distance_ > 0)
    truncateDepth(depths_curr_[0], max_icp_distance_);

  // downsample map for each pyramid level
  for (int i = 1; i < LEVELS; ++i)
    pyrDown (depths_curr_[i-1], depths_curr_[i]);

  // create vertex and normal maps for each pyramid level
  for (int i = 0; i < LEVELS; ++i)
  {
    createVMap (cam_intrinsics(i), depths_curr_[i], vmaps_curr_[i]);
    computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
  }
  
}

inline void
pcl::gpu::kinfuLS::KinfuTracker::saveCurrentMaps()
{
  Matrix3frm rot_id = Eigen::Matrix3f::Identity ();
  Mat33 identity_rotation = device_cast<Mat33> (rot_id);

  // save vertex and normal maps for each level, keeping camera coordinates (i.e. no transform)
  for (int i = 0; i < LEVELS; ++i)
  {
   transformMaps (vmaps_curr_[i], nmaps_curr_[i],identity_rotation, make_float3(0,0,0), vmaps_prev_[i], nmaps_prev_[i]);
  }  
}

inline bool 
pcl::gpu::kinfuLS::KinfuTracker::performICP(const Intr& cam_intrinsics, Matrix3frm& previous_global_rotation, Vector3f& previous_global_translation, Matrix3frm& current_global_rotation , Vector3f& current_global_translation)
{
  
  if(disable_icp_)
  {
    lost_=true;
    return (false);
  }
  
  // Compute inverse rotation
  Matrix3frm previous_global_rotation_inv = previous_global_rotation.inverse ();  // Rprev.t();  
 
 ///////////////////////////////////////////////
  // Convert pose to device type
  Mat33  device_cam_rot_local_prev_inv; 
  float3 device_cam_trans_local_prev;
  convertTransforms(previous_global_rotation_inv, previous_global_translation, device_cam_rot_local_prev_inv, device_cam_trans_local_prev);  
  device_cam_trans_local_prev -= getCyclicalBufferStructure ()->origin_metric; ;
 
  // Use temporary pose, so that we modify the current global pose only if ICP converged
  Matrix3frm resulting_rotation;
  Vector3f resulting_translation;
  
  // Initialize output pose to current pose
  current_global_rotation = previous_global_rotation;
  current_global_translation = previous_global_translation;
 
  ///////////////////////////////////////////////
  // Run ICP
  {
    //ScopeTime time("icp-all");
    for (int level_index = LEVELS-1; level_index>=0; --level_index)
    {
      int iter_num = icp_iterations_[level_index];
      
      // current vertex and normal maps
      MapArr& vmap_curr = vmaps_curr_[level_index];
      MapArr& nmap_curr = nmaps_curr_[level_index];   
      
      // previous vertex and normal maps
      MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
      MapArr& nmap_g_prev = nmaps_g_prev_[level_index];
      
      // We need to transform the maps from global to local coordinates
      Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); // Identity Rotation Matrix. Because we only need translation
      float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
      cube_origin = -cube_origin;
      
      MapArr& vmap_temp = vmap_g_prev;
      MapArr& nmap_temp = nmap_g_prev;
      transformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmap_g_prev, nmap_g_prev); 
      
      // run ICP for iter_num iterations (return false when lost)
      for (int iter = 0; iter < iter_num; ++iter)
      {
        //CONVERT POSES TO DEVICE TYPES
        // CURRENT LOCAL POSE
        Mat33    device_current_rotation = device_cast<Mat33> (current_global_rotation); // We do not deal with changes in rotations        
        float3 device_current_translation_local = device_cast<float3> (current_global_translation);
        device_current_translation_local -= getCyclicalBufferStructure ()->origin_metric; 
                
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
        Eigen::Matrix<double, 6, 1> b;

        // call the ICP function (see paper by Kok-Lim Low "Linear Least-squares Optimization for Point-to-Plane ICP Surface Registration")
        estimateCombined (device_current_rotation, device_current_translation_local, vmap_curr, nmap_curr, device_cam_rot_local_prev_inv, device_cam_trans_local_prev, cam_intrinsics (level_index), 
                          vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());

        // checking nullspace 
        double det = A.determinant ();
    
        if ( fabs (det) < 100000 /*1e-15*/ || pcl_isnan (det) ) //TODO find a threshold that makes ICP track well, but prevents it from generating wrong transforms
        {
          if (pcl_isnan (det)) cout << "qnan" << endl;
          if(lost_ == false)
            PCL_ERROR ("ICP LOST... PLEASE COME BACK TO THE LAST VALID POSE (green)\n");
          //reset (); //GUI will now show the user that ICP is lost. User needs to press "R" to reset the volume
          lost_ = true;
          return (false);
        }

        Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
        float alpha = result (0);
        float beta  = result (1);
        float gamma = result (2);

        // deduce incremental rotation and translation from ICP's results
        Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
        Vector3f cam_trans_incremental = result.tail<3> ();

        //compose global pose
        current_global_translation = cam_rot_incremental * current_global_translation + cam_trans_incremental;
        current_global_rotation = cam_rot_incremental * current_global_rotation;
      }
    }
  }
  // ICP has converged
  lost_ = false;
  return (true);
}

inline bool 
pcl::gpu::kinfuLS::KinfuTracker::performPairWiseICP(const Intr cam_intrinsics, Matrix3frm& resulting_rotation , Vector3f& resulting_translation)
{ 
  // we assume that both v and n maps are in the same coordinate space
  // initialize rotation and translation to respectively identity and zero
  Matrix3frm previous_rotation = Eigen::Matrix3f::Identity ();
  Matrix3frm previous_rotation_inv = previous_rotation.inverse ();
  Vector3f previous_translation = Vector3f(0,0,0);
  
 ///////////////////////////////////////////////
  // Convert pose to device type
  Mat33  device_cam_rot_prev_inv; 
  float3 device_cam_trans_prev;
  convertTransforms(previous_rotation_inv, previous_translation, device_cam_rot_prev_inv, device_cam_trans_prev);  
 
  // Initialize output pose to current pose (i.e. identity and zero translation)
  Matrix3frm current_rotation = previous_rotation;
  Vector3f current_translation = previous_translation;
   
  ///////////////////////////////////////////////
  // Run ICP
  {
    //ScopeTime time("icp-all");
    for (int level_index = LEVELS-1; level_index>=0; --level_index)
    {
      int iter_num = icp_iterations_[level_index];
      
      // current vertex and normal maps
      MapArr& vmap_curr = vmaps_curr_[level_index];
      MapArr& nmap_curr = nmaps_curr_[level_index];   
      
      // previous vertex and normal maps
      MapArr& vmap_prev = vmaps_prev_[level_index];
      MapArr& nmap_prev = nmaps_prev_[level_index];

      // no need to transform maps from global to local since they are both in camera coordinates
      
      // run ICP for iter_num iterations (return false when lost)
      for (int iter = 0; iter < iter_num; ++iter)
      {
        //CONVERT POSES TO DEVICE TYPES
        // CURRENT LOCAL POSE
        Mat33  device_current_rotation = device_cast<Mat33> (current_rotation);      
        float3 device_current_translation_local = device_cast<float3> (current_translation);
                
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
        Eigen::Matrix<double, 6, 1> b;

        // call the ICP function (see paper by Kok-Lim Low "Linear Least-squares Optimization for Point-to-Plane ICP Surface Registration")
        estimateCombined (device_current_rotation, device_current_translation_local, vmap_curr, nmap_curr, device_cam_rot_prev_inv, device_cam_trans_prev, cam_intrinsics (level_index), 
                          vmap_prev, nmap_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());

        // checking nullspace 
        double det = A.determinant ();
        
        if ( fabs (det) < 1e-15 || pcl_isnan (det) )
        {
          if (pcl_isnan (det)) cout << "qnan" << endl;
                    
          PCL_WARN ("ICP PairWise LOST...\n");
          //reset ();
          return (false);
        }

        Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
        float alpha = result (0);
        float beta  = result (1);
        float gamma = result (2);

        // deduce incremental rotation and translation from ICP's results
        Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
        Vector3f cam_trans_incremental = result.tail<3> ();

        //compose global pose
        current_translation = cam_rot_incremental * current_translation + cam_trans_incremental;
        current_rotation = cam_rot_incremental * current_rotation;
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // since raw depthmaps are quite noisy, we make sure the estimated transform is big enought to be taken into account
  float rnorm = rodrigues2(current_rotation).norm();
  float tnorm = (current_translation).norm();    
  const float alpha = 1.f;
  bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_ * 2.0f;
  
  if(integrate)
  {
    resulting_rotation = current_rotation;
    resulting_translation = current_translation;
  }
  else
  {
    resulting_rotation = Eigen::Matrix3f::Identity ();
    resulting_translation = Vector3f(0,0,0);
  }
  // ICP has converged
  return (true);
}

bool
pcl::gpu::kinfuLS::KinfuTracker::operator() (const DepthMap& depth_raw)
{ 
  // Intrisics of the camera
  Intr intr (fx_, fy_, cx_, cy_);
  
  // Physical volume size (meters)
  float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
  
  // process the incoming raw depth map
  prepareMaps (depth_raw, intr);
  
  // sync GPU device
  pcl::device::kinfuLS::sync ();
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Initialization at first frame
  if (global_time_ == 0) // this is the frist frame, the tsdf volume needs to be initialized
  {  
    // Initial rotation
    Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera
    Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
    // Initial translation
    Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose
    
    // Convert pose to device types
    Mat33 device_initial_cam_rot, device_initial_cam_rot_inv;
    float3 device_initial_cam_trans;
    convertTransforms (initial_cam_rot, initial_cam_rot_inv, initial_cam_trans, device_initial_cam_rot, device_initial_cam_rot_inv, device_initial_cam_trans);

    // integrate current depth map into tsdf volume, from default initial pose.
    integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
    
    // transform vertex and normal maps for each pyramid level
    for (int i = 0; i < LEVELS; ++i)
      transformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

    if(perform_last_scan_)
      finished_ = true;
      
    ++global_time_;
    
    // save current vertex and normal maps
    saveCurrentMaps ();    
    // return and wait for next frame
    return (false);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Iterative Closest Point
  // Get the last-known pose
  Matrix3frm last_known_global_rotation = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
  Vector3f   last_known_global_translation = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose
  // Declare variables to host ICP results 
  Matrix3frm current_global_rotation;
  Vector3f current_global_translation;
  // Call ICP
  if(!performICP(intr, last_known_global_rotation, last_known_global_translation, current_global_rotation, current_global_translation))
  {
    // ICP based on synthetic maps failed -> try to estimate the current camera pose based on previous and current raw maps
    Matrix3frm delta_rotation;
    Vector3f delta_translation;    
    if(!performPairWiseICP(intr, delta_rotation, delta_translation))
    {
      // save current vertex and normal maps
      saveCurrentMaps ();
      return (false);
    }
    
    // Pose estimation succeeded -> update estimated pose
    last_estimated_translation_ = delta_rotation * last_estimated_translation_ + delta_translation;
    last_estimated_rotation_ = delta_rotation * last_estimated_rotation_;
    // save current vertex and normal maps
    saveCurrentMaps ();
    return (true);
  }
  else
  {
    // ICP based on synthetic maps succeeded
    // Save newly-computed pose
    rmats_.push_back (current_global_rotation); 
    tvecs_.push_back (current_global_translation);
    // Update last estimated pose to current pairwise ICP result
    last_estimated_translation_ = current_global_translation;
    last_estimated_rotation_ = current_global_rotation;
  }  

  ///////////////////////////////////////////////////////////////////////////////////////////  
  // check if we need to shift
  has_shifted_ = cyclical_.checkForShift(tsdf_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_); // TODO make target distance from camera a param
  if(has_shifted_)
    PCL_WARN ("SHIFTING\n");
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // get the NEW local pose as device types
  Mat33  device_current_rotation_inv, device_current_rotation;   
  float3 device_current_translation_local;   
  Matrix3frm cam_rot_local_curr_inv = current_global_rotation.inverse (); //rotation (local = global)
  convertTransforms(cam_rot_local_curr_inv, current_global_rotation, current_global_translation, device_current_rotation_inv, device_current_rotation, device_current_translation_local); 
  device_current_translation_local -= getCyclicalBufferStructure()->origin_metric;   // translation (local translation = global translation - origin of cube)
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Integration check - We do not integrate volume if camera does not move far enought.  
  {
    float rnorm = rodrigues2(current_global_rotation.inverse() * last_known_global_rotation).norm();
    float tnorm = (current_global_translation - last_known_global_translation).norm();    
    const float alpha = 1.f;
    bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Volume integration  
    if (integrate)
    {
      integrateTsdfVolume (depth_raw, intr, device_volume_size, device_current_rotation_inv, device_current_translation_local, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Ray casting
  {          
    // generate synthetic vertex and normal maps from newly-found pose.
    raycast (intr, device_current_rotation, device_current_translation_local, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);
    
    // POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
    Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we never rotate our volume
    float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
    MapArr& vmap_temp = vmaps_g_prev_[0];
    MapArr& nmap_temp = nmaps_g_prev_[0];
    transformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);
    
    // create pyramid levels for vertex and normal maps
    for (int i = 1; i < LEVELS; ++i)
    {
      resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
      resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
    }
    pcl::device::kinfuLS::sync ();
  }

  if(has_shifted_ && perform_last_scan_)
    extractAndSaveWorld ();

    
  ++global_time_;
  // save current vertex and normal maps
  saveCurrentMaps ();
  return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::kinfuLS::KinfuTracker::getCameraPose (int time) const
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  Eigen::Affine3f aff;
  aff.linear () = rmats_[time];
  aff.translation () = tvecs_[time];
  return (aff);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::kinfuLS::KinfuTracker::getLastEstimatedPose () const
{
  Eigen::Affine3f aff;
  aff.linear () = last_estimated_rotation_;
  aff.translation () = last_estimated_translation_;
  return (aff);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
size_t
pcl::gpu::kinfuLS::KinfuTracker::getNumberOfPoses () const
{
  return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const pcl::gpu::kinfuLS::TsdfVolume& 
pcl::gpu::kinfuLS::KinfuTracker::volume() const 
{ 
  return *tsdf_volume_; 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::kinfuLS::TsdfVolume& 
pcl::gpu::kinfuLS::KinfuTracker::volume()
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const pcl::gpu::kinfuLS::ColorVolume& 
pcl::gpu::kinfuLS::KinfuTracker::colorVolume() const
{
  return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::kinfuLS::ColorVolume& 
pcl::gpu::kinfuLS::KinfuTracker::colorVolume()
{
  return *color_volume_;
}
     
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getImage (View& view) const
{
  //Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
  Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size () - 1];

  LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);

  view.create (rows_, cols_);
  generateImage (vmaps_g_prev_[0], nmaps_g_prev_[0], light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  convert (vmaps_g_prev_[0], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::kinfuLS::KinfuTracker::getLastFrameNormals (DeviceArray2D<NormalType>& normals) const
{
  normals.create (rows_, cols_);
  DeviceArray2D<float8>& n = (DeviceArray2D<float8>&)normals;
  convert (nmaps_g_prev_[0], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::kinfuLS::KinfuTracker::initColorIntegration(int max_weight)
{     
  color_volume_ = pcl::gpu::kinfuLS::ColorVolume::Ptr( new ColorVolume(*tsdf_volume_, max_weight) );  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool 
pcl::gpu::kinfuLS::KinfuTracker::operator() (const DepthMap& depth, const View& colors)
{ 
  bool res = (*this)(depth);

  if (res && color_volume_)
  {
    const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
    Intr intr(fx_, fy_, cx_, cy_);

    Matrix3frm R_inv = rmats_.back().inverse();
    Vector3f   t     = tvecs_.back();
    
    Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv);
    float3& device_tcurr = device_cast<float3> (t);
    
    updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0], 
        colors, device_volume_size, color_volume_->data(), color_volume_->getMaxWeight());
  }

  return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
  namespace gpu
  {
    namespace kinfuLS
    {
      PCL_EXPORTS void 
      paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
      {
        pcl::device::kinfuLS::paint3DView(rgb24, view, colors_weight);
      }

      PCL_EXPORTS void
      mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output)
      {
        const size_t size = min(cloud.size(), normals.size());
        output.create(size);

        const DeviceArray<float4>& c = (const DeviceArray<float4>&)cloud;
        const DeviceArray<float8>& n = (const DeviceArray<float8>&)normals;
        const DeviceArray<float12>& o = (const DeviceArray<float12>&)output;
        pcl::device::kinfuLS::mergePointNormal(c, n, o);           
      }

      Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
      {
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
        Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

        double rx = R(2, 1) - R(1, 2);
        double ry = R(0, 2) - R(2, 0);
        double rz = R(1, 0) - R(0, 1);

        double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
        double c = (R.trace() - 1) * 0.5;
        c = c > 1. ? 1. : c < -1. ? -1. : c;

        double theta = acos(c);

        if( s < 1e-5 )
        {
          double t;

          if( c > 0 )
            rx = ry = rz = 0;
          else
          {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
              rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
          }
        }
        else
        {
          double vth = 1/(2*s);
          vth *= theta;
          rx *= vth; ry *= vth; rz *= vth;
        }
        return Eigen::Vector3d(rx, ry, rz).cast<float>();
      }
    }
  }
}


//SPCL Stuff:

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
	pcl::gpu::kinfuLS::KinfuTracker::rgbdodometry(
	const cv::Mat& image0, const cv::Mat& _depth0, const cv::Mat& validMask0,
	const cv::Mat& image1, const cv::Mat& _depth1, const cv::Mat& validMask1,
	const cv::Mat& cameraMatrix, float minDepth, float maxDepth, float maxDepthDiff,
	const std::vector<int>& iterCounts, const std::vector<float>& minGradientMagnitudes,
	const DepthMap& depth_raw, const View * pcolor, FramedTransformation * frame_ptr
	)
{
	//ScopeTime time( "Kinfu Tracker All" );
	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->ResetFlag ) ) {
		reset();
		if ( frame_ptr->type_ == frame_ptr->DirectApply )
		{
			Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
			rmats_[0] = aff_rgbd.linear();
			tvecs_[0] = aff_rgbd.translation();
		}
	}

	device::kinfuLS::Intr intr (fx_, fy_, cx_, cy_, max_integrate_distance_);

	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreRegistrationFlag ) ) {
	}
	else
	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		//depth_raw.copyTo(depths_curr_[0]);
		device::kinfuLS::bilateralFilter (depth_raw, depths_curr_[0]);

		if (max_icp_distance_ > 0)
			device::kinfuLS::truncateDepth(depths_curr_[0], max_icp_distance_);

		for (int i = 1; i < LEVELS; ++i)
			device::kinfuLS::pyrDown (depths_curr_[i-1], depths_curr_[i]);

		for (int i = 0; i < LEVELS; ++i)
		{
			device::kinfuLS::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
			//device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
			computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}

		pcl::device::kinfuLS::sync ();
	}

	//can't perform more on first frame
	if (global_time_ == 0)
	{
		Matrix3frm initial_cam_rot = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Matrix3frm initial_cam_rot_inv = initial_cam_rot.inverse ();
		Vector3f   initial_cam_trans = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

		Mat33&  device_initial_cam_rot = device_cast<Mat33> (initial_cam_rot);
		Mat33&  device_initial_cam_rot_inv = device_cast<Mat33> (initial_cam_rot_inv);
		float3& device_initial_cam_trans = device_cast<float3>(initial_cam_trans);

		float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

		device::kinfuLS::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);
		//device::integrateTsdfVolume(depths_curr_[ 0 ], intr, device_volume_size, device_initial_cam_rot_inv, device_initial_cam_trans, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), getCyclicalBufferStructure (), depthRawScaled_);

		for (int i = 0; i < LEVELS; ++i)
			transformMaps (vmaps_curr_[i], nmaps_curr_[i], device_initial_cam_rot, device_initial_cam_trans, vmaps_g_prev_[i], nmaps_g_prev_[i]);

		if(perform_last_scan_)
			finished_ = true;
		++global_time_;
		return (false);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	// Iterative Closest Point

	// GET PREVIOUS GLOBAL TRANSFORM
	// Previous global rotation
	const int sobelSize = 3;
	const double sobelScale = 1./8;

	Mat depth0 = _depth0.clone(),
		depth1 = _depth1.clone();

	// check RGB-D input data
	CV_Assert( !image0.empty() );
	CV_Assert( image0.type() == CV_8UC1 );
	CV_Assert( depth0.type() == CV_32FC1 && depth0.size() == image0.size() );

	CV_Assert( image1.size() == image0.size() );
	CV_Assert( image1.type() == CV_8UC1 );
	CV_Assert( depth1.type() == CV_32FC1 && depth1.size() == image0.size() );

	// check masks
	CV_Assert( validMask0.empty() || (validMask0.type() == CV_8UC1 && validMask0.size() == image0.size()) );
	CV_Assert( validMask1.empty() || (validMask1.type() == CV_8UC1 && validMask1.size() == image0.size()) );

	// check camera params
	CV_Assert( cameraMatrix.type() == CV_32FC1 && cameraMatrix.size() == Size(3,3) );

	// other checks
	CV_Assert( iterCounts.empty() || minGradientMagnitudes.empty() ||
		minGradientMagnitudes.size() == iterCounts.size() );

	vector<int> defaultIterCounts;
	vector<float> defaultMinGradMagnitudes;
	vector<int> const* iterCountsPtr = &iterCounts;
	vector<float> const* minGradientMagnitudesPtr = &minGradientMagnitudes;
	if( iterCounts.empty() || minGradientMagnitudes.empty() )
	{
		defaultIterCounts.resize(4);
		defaultIterCounts[0] = 7;
		defaultIterCounts[1] = 7;
		defaultIterCounts[2] = 7;
		defaultIterCounts[3] = 10;

		defaultMinGradMagnitudes.resize(4);
		defaultMinGradMagnitudes[0] = 12;
		defaultMinGradMagnitudes[1] = 5;
		defaultMinGradMagnitudes[2] = 3;
		defaultMinGradMagnitudes[3] = 1;

		iterCountsPtr = &defaultIterCounts;
		minGradientMagnitudesPtr = &defaultMinGradMagnitudes;
	}

	preprocessDepth( depth0, depth1, validMask0, validMask1, minDepth, maxDepth );

	vector<Mat> pyramidImage0, pyramidDepth0,
		pyramidImage1, pyramidDepth1, pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1,
		pyramidCameraMatrix;
	buildPyramids( image0, image1, depth0, depth1, cameraMatrix, sobelSize, sobelScale, *minGradientMagnitudesPtr,
		pyramidImage0, pyramidDepth0, pyramidImage1, pyramidDepth1,
		pyramid_dI_dx1, pyramid_dI_dy1, pyramidTexturedMask1, pyramidCameraMatrix );

	Mat resultRt = Mat::eye(4,4,CV_64FC1);
	Mat currRt, ksi;

	Matrix3frm cam_rot_global_prev = rmats_[global_time_ - 1];            // [Ri|ti] - pos of camera, i.e.
	// Previous global translation
	Vector3f   cam_trans_global_prev = tvecs_[global_time_ - 1];          // transform from camera to global coo space for (i-1)th camera pose

	if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->InitializeOnly ) ) {
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
		cam_rot_global_prev = aff_rgbd.linear();
		cam_trans_global_prev = aff_rgbd.translation();
	} else if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->IncrementalOnly ) ) {
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ * getCameraPose().matrix() );
		cam_rot_global_prev = aff_rgbd.linear();
		cam_trans_global_prev = aff_rgbd.translation();
	}


	// Previous global inverse rotation
	Matrix3frm cam_rot_global_prev_inv = cam_rot_global_prev.inverse ();  // Rprev.t();

	// GET CURRENT GLOBAL TRANSFORM
	Matrix3frm cam_rot_global_curr = cam_rot_global_prev;                 // transform to global coo for ith camera pose
	Vector3f   cam_trans_global_curr = cam_trans_global_prev;

	// CONVERT TO DEVICE TYPES 
	//LOCAL PREVIOUS TRANSFORM
	Mat33&  device_cam_rot_local_prev_inv = device_cast<Mat33> (cam_rot_global_prev_inv);
	Mat33&  device_cam_rot_local_prev = device_cast<Mat33> (cam_rot_global_prev); 

	float3& device_cam_trans_local_prev_tmp = device_cast<float3> (cam_trans_global_prev);
	float3 device_cam_trans_local_prev;
	device_cam_trans_local_prev.x = device_cam_trans_local_prev_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_prev.y = device_cam_trans_local_prev_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_prev.z = device_cam_trans_local_prev_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;
	float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

	///////////////////////////////////////////////////////////////////////////////////////////
	// Ray casting
	/*Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);*/
	{          
		//ScopeTime time( ">>> raycast" );
		raycast (intr, device_cam_rot_local_prev, device_cam_trans_local_prev, tsdf_volume_->getTsdfTruncDist (), device_volume_size, tsdf_volume_->data (), getCyclicalBufferStructure (), vmaps_g_prev_[0], nmaps_g_prev_[0]);    
	}
	{
		// POST-PROCESSING: We need to transform the newly raycasted maps into the global space.
		//ScopeTime time( ">>> transformation based on raycast" );
		Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); /// Identity Rotation Matrix. Because we only need translation
		float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;

		//~ PCL_INFO ("Raycasting with cube origin at %f, %f, %f\n", cube_origin.x, cube_origin.y, cube_origin.z);

		MapArr& vmap_temp = vmaps_g_prev_[0];
		MapArr& nmap_temp = nmaps_g_prev_[0];

		transformMaps (vmap_temp, nmap_temp, rotation_id, cube_origin, vmaps_g_prev_[0], nmaps_g_prev_[0]);

		for (int i = 1; i < LEVELS; ++i)
		{
			resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
			resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
		}
		pcl::device::kinfuLS::sync ();
	}

	if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreRegistrationFlag ) )
	{
		rmats_.push_back (cam_rot_global_prev); 
		tvecs_.push_back (cam_trans_global_prev);
	}
	else if ( frame_ptr != NULL && ( frame_ptr->type_ == frame_ptr->DirectApply ) )
	{
		//Eigen::Affine3f aff_rgbd( getCameraPose( 0 ).matrix() * frame_ptr->transformation_ );
		Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
		cam_rot_global_curr = aff_rgbd.linear();
		cam_trans_global_curr = aff_rgbd.translation();
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}
	else
	{
		//ScopeTime time(">>> icp-all");

		Eigen::Matrix4f result_trans = Eigen::Matrix4f::Identity();
		Eigen::Affine3f mat_prev;
		mat_prev.linear() = cam_rot_global_prev;
		mat_prev.translation() = cam_trans_global_prev;

		for( int level_index = (int)iterCountsPtr->size() - 1; level_index >= 0; level_index-- )
		{
			// We need to transform the maps from global to the local coordinates
			Mat33&  rotation_id = device_cast<Mat33> (rmats_[0]); // Identity Rotation Matrix. Because we only need translation
			float3 cube_origin = (getCyclicalBufferStructure ())->origin_metric;
			cube_origin.x = -cube_origin.x;
			cube_origin.y = -cube_origin.y;
			cube_origin.z = -cube_origin.z;

			const Mat& levelCameraMatrix = pyramidCameraMatrix[ level_index ];

			const Mat& levelImage0 = pyramidImage0[ level_index ];
			const Mat& levelDepth0 = pyramidDepth0[ level_index ];
			Mat levelCloud0;
			cvtDepth2Cloud( pyramidDepth0[ level_index ], levelCloud0, levelCameraMatrix );

			const Mat& levelImage1 = pyramidImage1[ level_index ];
			const Mat& levelDepth1 = pyramidDepth1[ level_index ];
			const Mat& level_dI_dx1 = pyramid_dI_dx1[ level_index ];
			const Mat& level_dI_dy1 = pyramid_dI_dy1[ level_index ];

			CV_Assert( level_dI_dx1.type() == CV_16S );
			CV_Assert( level_dI_dy1.type() == CV_16S );

			const double fx = levelCameraMatrix.at<double>(0,0);
			const double fy = levelCameraMatrix.at<double>(1,1);
			const double determinantThreshold = 1e-6;

			Mat corresps( levelImage0.size(), levelImage0.type() );

			for( int iter = 0; iter < (*iterCountsPtr)[ level_index ]; iter ++ ) {
				bool odo_good = true;

				int correspsCount = computeCorresp( levelCameraMatrix, levelCameraMatrix.inv(), resultRt.inv(DECOMP_SVD),
					levelDepth0, levelDepth1, pyramidTexturedMask1[ level_index ], maxDepthDiff,
					corresps );
				if( correspsCount == 0 ) {
					odo_good = false;
				} else {
					odo_good = computeKsi( 0,
						levelImage0, levelCloud0,
						levelImage1, level_dI_dx1, level_dI_dy1,
						corresps, correspsCount,
						fx, fy, sobelScale, determinantThreshold,
						ksi, AA_, bb_ );
				}

				if ( odo_good == false ) {
					AA_.setZero();
					bb_.setZero();
				}

				Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr);/// We have not dealt with changes in rotations

				float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
				float3 device_cam_trans_local_curr; 
				device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
				device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
				device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;

				AAA_ = AA_;
				bbb_ = bb_;

				double det = AAA_.determinant ();

				if ( fabs (det) < 1e-15 || pcl_isnan (det) )
				{
					if (pcl_isnan (det)) cout << "qnan" << endl;

					PCL_ERROR ("LOST ... @%d frame.%d level.%d iteration, matrices are\n", global_time_, level_index, iter);
					cout << "Determinant : " << det << endl;
					cout << "Singular matrix :" << endl << A_ << endl;
					cout << "Corresponding b :" << endl << b_ << endl;

					if ( frame_ptr != NULL && frame_ptr->type_ == frame_ptr->InitializeOnly ) {
						Eigen::Affine3f aff_rgbd( frame_ptr->transformation_ );
						cam_rot_global_curr = aff_rgbd.linear();
						cam_trans_global_curr = aff_rgbd.translation();
						break;
					} else {
						reset ();
						return (false);
					}
				}

				Eigen::Matrix< float, 6, 1 > result = AAA_.llt().solve( bbb_ ).cast< float >();

				float alpha = result (0);
				float beta  = result (1);
				float gamma = result (2);

				Eigen::Matrix3f cam_rot_incremental = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
				Vector3f cam_trans_incremental = result.tail<3> ();

				Eigen::Affine3f mat_inc;
				mat_inc.linear() = cam_rot_incremental;
				mat_inc.translation() = cam_trans_incremental;

				result_trans = mat_inc * result_trans;
				Eigen::Matrix4d temp = result_trans.cast< double >();
				eigen2cv( temp, resultRt );

				Eigen::Affine3f mat_curr;
				mat_curr.matrix() = mat_prev.matrix() * result_trans;

				cam_rot_global_curr = mat_curr.linear();
				cam_trans_global_curr = mat_curr.translation();
			}
		}

		//save tranform
		rmats_.push_back (cam_rot_global_curr); 
		tvecs_.push_back (cam_trans_global_curr);
	}

/*
	bool has_shifted = false;
	if ( force_shift_ ) {
		has_shifted = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, true, perform_last_scan_, force_shift_, extract_world_);
		force_shift_ = false;
	} else {
		force_shift_ = cyclical_.checkForShift(tsdf_volume_, color_volume_, getCameraPose (), 0.6 * volume_size_, false, perform_last_scan_, force_shift_, extract_world_);
	}

	if(has_shifted)
		PCL_WARN ("SHIFTING\n");
  */  //As there are no other references to it, it's safe to assume it's always false


	// get NEW local rotation 
	Matrix3frm cam_rot_local_curr_inv = cam_rot_global_curr.inverse ();
	Mat33&  device_cam_rot_local_curr_inv = device_cast<Mat33> (cam_rot_local_curr_inv);
	Mat33&  device_cam_rot_local_curr = device_cast<Mat33> (cam_rot_global_curr); 

	// get NEW local translation
	float3& device_cam_trans_local_curr_tmp = device_cast<float3> (cam_trans_global_curr);
	float3 device_cam_trans_local_curr;
	device_cam_trans_local_curr.x = device_cam_trans_local_curr_tmp.x - (getCyclicalBufferStructure ())->origin_metric.x;
	device_cam_trans_local_curr.y = device_cam_trans_local_curr_tmp.y - (getCyclicalBufferStructure ())->origin_metric.y;
	device_cam_trans_local_curr.z = device_cam_trans_local_curr_tmp.z - (getCyclicalBufferStructure ())->origin_metric.z;  


	///////////////////////////////////////////////////////////////////////////////////////////
	// Integration check - We do not integrate volume if camera does not move.  
	float rnorm = rodrigues2(cam_rot_global_curr.inverse() * cam_rot_global_prev).norm();
	float tnorm = (cam_trans_global_curr - cam_trans_global_prev).norm();    
	const float alpha = 1.f;
	bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
	integrate = true;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration
	if (integrate)
	{
		//integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
		if ( frame_ptr != NULL && ( frame_ptr->flag_ & frame_ptr->IgnoreIntegrationFlag ) ) {
		} else {
			//ScopeTime time( ">>> integrate" );
			integrateTsdfVolume (depth_raw, intr, device_volume_size, device_cam_rot_local_curr_inv, device_cam_trans_local_curr, tsdf_volume_->getTsdfTruncDist (), tsdf_volume_->data (), getCyclicalBufferStructure (), depthRawScaled_);
		}
	}

	{
		if ( pcolor && color_volume_ )
		{
			//ScopeTime time( ">>> update color" );
			const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

			device::kinfuLS::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_cam_rot_local_curr_inv, device_cam_trans_local_curr, vmaps_g_prev_[0], 
				*pcolor, device_volume_size, color_volume_->data(), getCyclicalBufferStructure(), color_volume_->getMaxWeight());
		}
	}

	++global_time_;
	return (true);
}

bool
	pcl::gpu::kinfuLS::KinfuTracker::intersect( int bounds[ 6 ] )
{
	pcl::gpu::kinfuLS::tsdf_buffer* buffer = getCyclicalBufferStructure();
	// need more work here.
	return true;
}

