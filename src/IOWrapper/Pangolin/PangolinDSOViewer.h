/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include <pangolin/pangolin.h>
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"
#include <map>
#include <deque>
#include "util/ImageAndExposure.h"
#include <pangolin/gl/glfont.h>
#include <pangolin/gl/gltext.h>

#include <vector>
#include <string>
namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class KeyFrameDisplay;

struct GraphConnection
{
	KeyFrameDisplay* from;
	KeyFrameDisplay* to;
	int fwdMarg, bwdMarg, fwdAct, bwdAct;
};


class PangolinDSOViewer : public Output3DWrapper
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PangolinDSOViewer(int w, int h, bool startRunThread=true, std::string source="sample");
	virtual ~PangolinDSOViewer();

	// returns the selected keyframe if it is the first time the keyframe was requested, else return -1
	virtual int getselectedkf() override;

	// track if the select4ed keyframe as been requested before
	bool selectedkfchange;

	// the keyframe clicked by the user
	int selectedkf;

	void run();
	void close();

	void addImageToDisplay(std::string name, MinimalImageB3* image);
	void clearAllImagesToDisplay();


	// ==================== Output3DWrapper Functionality ======================
    virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override;
virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override;
    virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override;

	// like the original pushLiveFrame but has color
    virtual void pushLiveFrame(ColorImageAndExposure* image) override;
    virtual void pushLiveFrame(FrameHessian* image) override;

	// push the selected keyframe's color image 
    virtual void pushRequestedFrame(ColorImageAndExposure* image) override;
    virtual void pushDepthImage(MinimalImageB3* image) override;
    virtual bool needPushDepthImage() override;

    virtual void join() override;

    virtual void reset() override;
private:
	enum PlaybackMode { PAUSE, FORWARD, REVERSE };
	enum CompassMode { ANGLE, POSITION,NOTHING };
	enum Direction { UP, LEFT,DOWN,RIGHT };
	enum CloseLoop { CL_OFF,CL_SELECT_FIRST,CL_SELECT_SECOND,CL_CONFIRM };
	bool needReset;
	void reset_internal();
	void drawConstraints();
	void drawCircle(float cx, float cy, float cz, float r, int num_segments);
	void drawArc(Sophus::Vector3f p1,Sophus::Vector3f p2,int angle);
	void drawCompass(float cx, float cy, float cz, float r, int angle, int num_segments, int pointerScale, bool ringSelected,bool centreSelected);
	void angleglVertex3f(float cx,float dx, float cy, float cz, float dz, float theta);
	void drawAbsSphere(float ax, float ay, float az, double r, int lats, int longs);
	boost::thread runThread;
	bool running;
	int w,h;
	std::string filename;



	// images rendering
	boost::mutex openImagesMutex;
	MinimalImageB3* internalVideoImg;
	MinimalImageB3* internalVideoPlayerImg;
	MinimalImageB3* internalKFImg;
	MinimalImageB3* internalResImg;
	bool videoImgChanged, kfImgChanged, resImgChanged, videoPlayerImgChanged;



	// 3D model rendering
	boost::mutex model3DMutex;
	KeyFrameDisplay* currentCam;
	std::vector<KeyFrameDisplay*> keyframes;
	std::vector<Vec3f,Eigen::aligned_allocator<Vec3f>> allFramePoses;
	std::map<int, KeyFrameDisplay*> keyframesByKFID;
	std::vector<GraphConnection,Eigen::aligned_allocator<GraphConnection>> connections;



	// render settings
	bool settings_showKFCameras;
	bool settings_showCurrentCamera;
	bool settings_showTrajectory;
	bool settings_showFullTrajectory;
	bool settings_showActiveConstraints;
	bool settings_showAllConstraints;

	float settings_scaledVarTH;
	float settings_absVarTH;
	int settings_pointCloudMode;
	float settings_minRelBS;
	int settings_sparsity;


	// timings
	struct timeval last_track;
	struct timeval last_map;


	std::deque<float> lastNTrackingMs;
	std::deque<float> lastNMappingMs;
};



}



}
