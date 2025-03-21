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

#undef Success

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include "util/NumType.h"
#include <pangolin/pangolin.h>
#include "util/settings.h"

#include <sstream>
#include <fstream>
#include <vector>
namespace dso
{
class CalibHessian;
class FrameHessian;
class FrameShell;

namespace IOWrap
{

template<int ppp>
struct InputPointSparse
{
	float u;
	float v;
	float idpeth;
	float idepth_hessian;
	float relObsBaseline;
	int numGoodRes;
	unsigned char color[ppp];
	float pixelcolor[patternNum][3];
	unsigned char status;
};

struct MyVertex
{
	float point[3];
	unsigned char color[4];
};

// stores a pointcloud associated to a Keyframe.
class KeyFrameDisplay
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	KeyFrameDisplay();
	~KeyFrameDisplay();

	double timestamp; // copied from image

	// copies points from KF over to internal buffer,
	// keeping some additional information so we can render it differently.
	void setFromKF(FrameHessian* fh, CalibHessian* HCalib);

	// copies points from KF over to internal buffer,
	// keeping some additional information so we can render it differently.
	void setFromF(FrameShell* fs, CalibHessian* HCalib);

	// copies & filters internal data to GL buffer for rendering. if nothing to do: does nothing.
	bool refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity);

	// get heading of frame
	int  getCompass(float offsetAngle);

	// renders cam & pointcloud.
	void drawCam(float lineWidth = 1, int* color = 0, float sizeFactor=1);
	void drawPC(float pointSize,int* color=0);
	void drawIndexedPC(float pointSize);

	int id;
	bool active;
	SE3 camToWorld;
	SE3 worldToCam;

    inline bool operator < (const KeyFrameDisplay& other) const
    {
        return (id < other.id);
    }
	void addMarking(int x1,int x2, int y1, int y2,float scaledTH, float absTH, int mode, float minBS, int sparsity);
	std::vector<float> getMarkingSize(int x1, int x2, int y1, int y2);
	void removeMarking();
	// export pointcloud to PCL format
	void addPC(pcl::PointCloud<pcl::PointXYZRGB>* pcloud,float scaledTH, float absTH, int mode, float minBS, int sparsity);
	Vec3f getPCbyMatrix( GLdouble cursor_pos[3]);
	Vec3f getPCfromID(int id);
	Vec3f centerCoordinates;
	Vec3f getCamCenter();
	std::vector<float> getBounds();
	std::vector<double> getSquareError(float cx, float cz,float r);
	Sophus::Vector3f getCamCoords();

	void setCamOffset(Sophus::Vector3f offset);
	Sophus::Vector3f  getCamOffset();
private:
	Sophus::Matrix4f moffset;
	bool writingMutex;
	float fx,fy,cx,cy;
	float fxi,fyi,cxi,cyi;
	int width, height;

	float my_scaledTH, my_absTH, my_scale;
	int my_sparsifyFactor;
	int my_displayMode;
	float my_minRelBS;
	bool needRefresh;


	int numSparsePoints;
	int numSparseBufferSize;
    InputPointSparse<MAX_RES_PER_POINT>* originalInputSparse;
	std::vector<std::vector<int>> boxes;
	std::vector<std::vector<float>> boxDim;

	bool bufferValid;
	int numGLBufferPoints;
	int numGLBufferGoodPoints;
	pangolin::GlBuffer vertexBuffer;
	pangolin::GlBuffer colorBuffer;
	Vec3f *tmpVertexBufferIndex;
	int numPoints;
	float maxX,minX,maxY,minY,maxZ,minZ;
};

}
}

