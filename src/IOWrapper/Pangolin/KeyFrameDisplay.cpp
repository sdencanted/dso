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

#include <stdio.h>
#include <vector>
#include "util/settings.h"

//#include <GL/glx.h>
//#include <GL/gl.h>
//#include <GL/glu.h>

#include <pangolin/pangolin.h>
#include "KeyFrameDisplay.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"

namespace dso
{
	namespace IOWrap
	{

		KeyFrameDisplay::KeyFrameDisplay()
		{
			writingMutex = false;
			originalInputSparse = 0;
			numSparseBufferSize = 0;
			numSparsePoints = 0;

			id = 0;
			active = true;
			camToWorld = SE3();

			needRefresh = true;

			my_scaledTH = 1e10;
			my_absTH = 1e10;
			my_displayMode = 1;
			my_minRelBS = 0;
			my_sparsifyFactor = 1;

			numGLBufferPoints = 0;
			bufferValid = false;
		}
		void KeyFrameDisplay::setFromF(FrameShell *frame, CalibHessian *HCalib)
		{
			id = frame->id;
			fx = HCalib->fxl();
			fy = HCalib->fyl();
			cx = HCalib->cxl();
			cy = HCalib->cyl();
			width = wG[0];
			height = hG[0];
			fxi = 1 / fx;
			fyi = 1 / fy;
			cxi = -cx / fx;
			cyi = -cy / fy;
			camToWorld = frame->camToWorld;
			needRefresh = true;
		}

		void KeyFrameDisplay::setFromKF(FrameHessian *fh, CalibHessian *HCalib)
		{
			writingMutex = true;
			setFromF(fh->shell, HCalib);

			// add all traces, inlier and outlier points.
			int npoints = fh->immaturePoints.size() +
						  fh->pointHessians.size() +
						  fh->pointHessiansMarginalized.size() +
						  fh->pointHessiansOut.size();

			if (numSparseBufferSize < npoints)
			{
				if (originalInputSparse != 0)
					delete originalInputSparse;
				numSparseBufferSize = npoints + 100;
				originalInputSparse = new InputPointSparse<MAX_RES_PER_POINT>[numSparseBufferSize];
			}

			InputPointSparse<MAX_RES_PER_POINT> *pc = originalInputSparse;
			numSparsePoints = 0;
			for (ImmaturePoint *p : fh->immaturePoints)
			{
				for (int i = 0; i < patternNum; i++)
				{
					pc[numSparsePoints].color[i] = p->color[i]; // unused
					for (int u = 0; u < 3; u++)
						pc[numSparsePoints].pixelcolor[i][u] = p->pixelcolor[i][u];
				}

				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = (p->idepth_max + p->idepth_min) * 0.5f;
				pc[numSparsePoints].idepth_hessian = 1000;
				pc[numSparsePoints].relObsBaseline = 0;
				pc[numSparsePoints].numGoodRes = 1;
				pc[numSparsePoints].status = 0;
				numSparsePoints++;
			}

			for (PointHessian *p : fh->pointHessians)
			{
				for (int i = 0; i < patternNum; i++)
				{
					pc[numSparsePoints].color[i] = p->color[i]; // unused
					for (int u = 0; u < 3; u++)
						pc[numSparsePoints].pixelcolor[i][u] = p->pixelcolor[i][u];
				}
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = 1;

				numSparsePoints++;
			}

			for (PointHessian *p : fh->pointHessiansMarginalized)
			{
				for (int i = 0; i < patternNum; i++)
				{
					pc[numSparsePoints].color[i] = p->color[i];
					for (int u = 0; u < 3; u++)
						pc[numSparsePoints].pixelcolor[i][u] = p->pixelcolor[i][u];
				}
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = 2;
				numSparsePoints++;
			}

			for (PointHessian *p : fh->pointHessiansOut)
			{
				for (int i = 0; i < patternNum; i++)
				{
					pc[numSparsePoints].color[i] = p->color[i]; // unused
					for (int u = 0; u < 3; u++)
						pc[numSparsePoints].pixelcolor[i][u] = p->pixelcolor[i][u];
				}
				pc[numSparsePoints].u = p->u;
				pc[numSparsePoints].v = p->v;
				pc[numSparsePoints].idpeth = p->idepth_scaled;
				pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
				pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
				pc[numSparsePoints].numGoodRes = 0;
				pc[numSparsePoints].status = 3;
				numSparsePoints++;
			}
			assert(numSparsePoints <= npoints);

			camToWorld = fh->PRE_camToWorld;
			needRefresh = true;

			writingMutex = false;
		}

		KeyFrameDisplay::~KeyFrameDisplay()
		{
			if (originalInputSparse != 0)
				delete[] originalInputSparse;
		}
		std::vector<float> KeyFrameDisplay::getBounds()
		{
			std::vector<float> bounds;
			bounds.push_back(maxX);
			bounds.push_back(minX);
			bounds.push_back(maxY);
			bounds.push_back(minY);
			bounds.push_back(maxZ);
			bounds.push_back(minZ);
			return bounds;
		}
		bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity)
		{
			std::vector<float> camX, camY;
			std::vector<float> depthTotal;
			std::vector<float> allX, allY, allZ;
			maxX = maxY = maxZ = minX = minY = minZ = 0;
			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			Sophus::Vector4f tempVector;
			tempVector[3] = 1; //signifies the vector is a position in space

			if (canRefresh)
			{
				needRefresh = needRefresh ||
							  my_scaledTH != scaledTH ||
							  my_absTH != absTH ||
							  my_displayMode != mode ||
							  my_minRelBS != minBS ||
							  my_sparsifyFactor != sparsity;
			}
			if (writingMutex)
				return false;
			if (!needRefresh)
				return false;
			needRefresh = false;

			my_scaledTH = scaledTH;
			my_absTH = absTH;
			my_displayMode = mode;
			my_minRelBS = minBS;
			my_sparsifyFactor = sparsity;

			// if there are no vertices, done!
			if (numSparsePoints == 0)
				return false;

			// make data
			Vec3f *tmpVertexBuffer = new Vec3f[numSparsePoints * patternNum * 2];
			Vec3b *tmpColorBuffer = new Vec3b[numSparsePoints * patternNum * 2];
			int vertexBufferNumPoints = 0;
			bool drawMarking = false;
			float bestScoreTopLeft = 99999;
			float bestScoreBotRight = 99999;
			float scoreTopLeft, scoreBotRight;
			std::vector<float> positionTopLeft(2), positionBotRight(2);
			std::vector<float> pixelTopLeft(2), pixelBotRight(2);
			pixelBotRight[0]=9999;
			pixelBotRight[1]=9999;

			std::vector<int> abox;
			bool doboxes = boxes.size() > boxDim.size();
			// printf("test\n");
			if (doboxes)
			{
				abox = boxes.back();
				if (abox[0] == 0 && abox[1] == 0 && abox[2] == 0 && abox[3] == 0)
					doboxes = false;
				else
					printf("box %d %d %d %d %d\n", abox[0], abox[1], abox[2], abox[3], abox[0] == 0);
			}
			for (int i = 0; i < numSparsePoints; i++)
			{
				// if (doboxes) printf("test0\n" );
				/* display modes:
		 * my_displayMode==0 - all pts, color-coded
		 * my_displayMode==1 - normal points
		 * my_displayMode==2 - active only
		 * my_displayMode==3 - nothing
		 */

				if (my_displayMode == 1 && originalInputSparse[i].status != 1 && originalInputSparse[i].status != 2)
					continue;
				// if (doboxes) printf("test1\n" );
				if (my_displayMode == 2 && originalInputSparse[i].status != 1)
					continue;
				// if (doboxes) printf("test2\n" );
				if (my_displayMode > 2)
					continue;
				// if (doboxes) printf("test3\n" );

				if (originalInputSparse[i].idpeth < 0)
					continue;
				// if (doboxes) printf("test4\n" );

				float depth = 1.0f / originalInputSparse[i].idpeth;
				depthTotal.push_back(depth);
				float depth4 = depth * depth;
				depth4 *= depth4;
				float var = (1.0f / (originalInputSparse[i].idepth_hessian + 0.01));

				if (var * depth4 > my_scaledTH)
					continue;

				// if (doboxes) printf("test5\n" );

				if (var > my_absTH)
					continue;

				// if (doboxes) printf("test6\n" );

				if (originalInputSparse[i].relObsBaseline < my_minRelBS)
				{
					// printf("bs %f\n",my_minRelBS);
					continue;
				}
				// printf ("sizes %d %d\n",boxes.size() , boxDim.size());
				if (doboxes)
				{
					// std::vector<int> box = boxes.back(); // add pixel to potential point for calculating width / height
					// boxDim;
					printf("aaaa %f\n",((originalInputSparse[i].u) * fxi + cxi) * depth);
					
					camX.insert(std::upper_bound(camX.begin(), camX.end(), ((originalInputSparse[i].u) * fxi + cxi) * depth), ((originalInputSparse[i].u) * fxi + cxi) * depth);

					camY.insert(std::upper_bound(camY.begin(), camY.end(), ((originalInputSparse[i].v) * fyi + cyi) * depth), ((originalInputSparse[i].v) * fyi + cyi) * depth);
					// scoreTopLeft = abs((int)abox[0] - (int)originalInputSparse[i].u) + abs((int)abox[2] - (int)originalInputSparse[i].v);
					// scoreBotRight = abs((int)abox[1] - (int)originalInputSparse[i].u) + abs((int)abox[3] - (int)originalInputSparse[i].v);

					// printf("scores %f %f\n", scoreTopLeft, scoreBotRight);
					// printf("position topleft %f %f\n", positionTopLeft[0], positionTopLeft[1]);
					// printf("position botright %f %f\n", positionBotRight[0], positionBotRight[1]);
					// printf("pixel topleft %f %f\n", pixelTopLeft[0], pixelTopLeft[1]);
					// printf("pixel botright %f %f\n", pixelBotRight[0], pixelBotRight[1]);
					// // if (scoreTopLeft>0 && scoreTopLeft < bestScoreTopLeft && (abox[1] >= originalInputSparse[i].u && abox[3] >= originalInputSparse[i].v))
					// if (scoreTopLeft>0 && scoreTopLeft < bestScoreTopLeft)
					// {
					// 	bestScoreTopLeft = scoreTopLeft;
					// 	printf("score\n");
					// 	positionTopLeft[0] = ((originalInputSparse[i].u) * fxi + cxi) * depth;
					// 	positionTopLeft[1] = ((originalInputSparse[i].v) * fyi + cyi) * depth;
					// 	pixelTopLeft[0] = originalInputSparse[i].u;
					// 	pixelTopLeft[1] = originalInputSparse[i].v;
					// 	// positionTopLeft[2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f));
					// }
					// // if (scoreBotRight>0 && scoreBotRight < bestScoreBotRight && (abox[0] <= originalInputSparse[i].u && abox[2] <= originalInputSparse[i].v))
					// if (scoreBotRight>0 && scoreBotRight < bestScoreBotRight)
					// {
					// 	bestScoreBotRight = scoreBotRight;
					// 	printf("score\n");

					// 	positionBotRight[0] = ((originalInputSparse[i].u) * fxi + cxi) * depth;
					// 	positionBotRight[1] = ((originalInputSparse[i].v) * fyi + cyi) * depth;
					// 	pixelBotRight[0] = originalInputSparse[i].u;
					// 	pixelBotRight[1] = originalInputSparse[i].v;
					// 	// positionBotRight[2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f));
					// }
				}
				drawMarking = false;
				for (auto bbox : boxes)
				{
					if (bbox[0] - wG[0] * 0.05 <= (int)originalInputSparse[i].u && bbox[1] + wG[0] * 0.05 >= (int)originalInputSparse[i].u && bbox[2] - hG[0] * 0.05 < (int)originalInputSparse[i].v && bbox[3] + hG[0] * 0.05 >= (int)originalInputSparse[i].v)
					{
						drawMarking = true;
						break;
					}
				}
				for (int pnt = 0; pnt < patternNum; pnt++)
				{
					if (my_sparsifyFactor > 1 && rand() % my_sparsifyFactor != 0)
						continue;
					int dx = patternP[pnt][0];
					int dy = patternP[pnt][1];

					tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u + dx) * fxi + cxi) * depth;
					tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v + dy) * fyi + cyi) * depth;
					tmpVertexBuffer[vertexBufferNumPoints][2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f));
					tempVector[0] = tmpVertexBuffer[vertexBufferNumPoints][0];
					tempVector[1] = tmpVertexBuffer[vertexBufferNumPoints][1];
					tempVector[2] = tmpVertexBuffer[vertexBufferNumPoints][2];
					tempVector = m * tempVector;
					// allX.push_back(tempVector[0]);
					// allY.push_back(tempVector[1]);
					// allZ.push_back(tempVector[2]);

					allX.insert(std::upper_bound(allX.begin(), allX.end(), tempVector[0]), tempVector[0]);
					allY.insert(std::upper_bound(allY.begin(), allY.end(), tempVector[1]), tempVector[1]);
					allZ.insert(std::upper_bound(allZ.begin(), allZ.end(), tempVector[2]), tempVector[2]);
					// maxX=std::max(maxX,tempVector[0]);
					// minX=std::min(minX,tempVector[0]);
					// maxY=std::max(maxY,tempVector[1]);
					// minY=std::min(minY,tempVector[1]);
					// maxZ=std::max(maxZ,tempVector[2]);
					// minZ=std::min(minZ,tempVector[2]);
					tmpColorBuffer[vertexBufferNumPoints][0] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][2]);
					tmpColorBuffer[vertexBufferNumPoints][1] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][1]);
					tmpColorBuffer[vertexBufferNumPoints][2] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][0]);
					if (drawMarking)
					{
						tmpVertexBuffer[vertexBufferNumPoints + 1][0] = ((originalInputSparse[i].u + dx) * fxi + cxi) * depth;
						tmpVertexBuffer[vertexBufferNumPoints + 1][1] = ((originalInputSparse[i].v + dy) * fyi + cyi) * depth;
						tmpVertexBuffer[vertexBufferNumPoints + 1][2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f)) - 0.01;

						tmpColorBuffer[vertexBufferNumPoints + 1][0] = (unsigned char)255;
						tmpColorBuffer[vertexBufferNumPoints + 1][1] = (unsigned char)0;
						tmpColorBuffer[vertexBufferNumPoints + 1][2] = (unsigned char)0;
					}

					vertexBufferNumPoints++;
					if (drawMarking)
						vertexBufferNumPoints++;

					assert(vertexBufferNumPoints <= numSparsePoints * patternNum * 2);
				}

				if (doboxes)
					printf("test\n");
				// std::sort(allX.begin(), allX.end());
				// std::sort(allY.begin(), allY.end());
				// std::sort(allZ.begin(), allZ.end());
				int percent5 = allX.size() * 0.05;
				int percent95 = allX.size() * 0.95;
				maxX = allX[percent95];
				minX = allX[percent5];
				maxY = allY[percent95];
				minY = allY[percent5];
				maxZ = allZ[percent95];
				minZ = allZ[percent5];
			}
			if (doboxes)
			{

				// if (bestScoreBotRight == 99999 || bestScoreTopLeft==99999)
				// { //failed{
				// 	printf("failed! %f %f\n", bestScoreBotRight, bestScoreTopLeft);
				// 	return false;
				// }
				// printf("passed! %f %f\n", bestScoreBotRight, bestScoreTopLeft);
				// printf("passed2! %d %d\n", abox[0], abox[2]);

				// int percentx5 = camX.size() * 0.05;
				// int percentx95 = camX.size() * 0.95;
				// int percenty5 = camY.size() * 0.05;
				// int percenty95 = camY.size() * 0.95;
				// printf("%d %d %d %d\n",percentx5,percentx95,percenty5,percenty95);
				std::vector<float> dims;
				// dims.push_back((camX[percentx95] - camX[percentx5]) * ((abox[1] - abox[0]) / wG[0]));
				// dims.push_back((camY[percenty95] - camY[percenty5]) * ((abox[3] - abox[2]) / hG[0]));
				if(depthTotal.size()>5){
					
					dims.push_back(((abox[1]-abox[0]) * fxi) * depthTotal[depthTotal.size()/2]);
					dims.push_back(((abox[3]-abox[2]) * fyi) * depthTotal[depthTotal.size()/2]);
					printf("pushing back %f %f\n", dims[0], dims[1]);
					boxDim.push_back(dims);
				}
			}
			if (vertexBufferNumPoints == 0)
			{
				delete[] tmpColorBuffer;
				delete[] tmpVertexBuffer;
				return true;
			}

			numGLBufferGoodPoints = vertexBufferNumPoints;
			if (numGLBufferGoodPoints > numGLBufferPoints)
			{
				numGLBufferPoints = vertexBufferNumPoints * 1.3;
				vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
				colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
			}
			vertexBuffer.Upload(tmpVertexBuffer, sizeof(float) * 3 * numGLBufferGoodPoints, 0);
			colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char) * 3 * numGLBufferGoodPoints, 0);
			bufferValid = true;
			delete[] tmpColorBuffer;
			delete[] tmpVertexBuffer;

			return true;
		}

		void drawSphere(double r, int lats, int longs)
		{
			int i, j;
			for (i = 0; i <= lats; i++)
			{
				double lat0 = M_PI * (-0.5 + (double)(i - 1) / lats);
				double z0 = sin(lat0);
				double zr0 = cos(lat0);

				double lat1 = M_PI * (-0.5 + (double)i / lats);
				double z1 = sin(lat1);
				double zr1 = cos(lat1);

				glBegin(GL_QUAD_STRIP);
				for (j = 0; j <= longs; j++)
				{
					double lng = 2 * M_PI * (double)(j - 1) / longs;
					double x = cos(lng);
					double y = sin(lng);

					glNormal3f(x * zr0, y * zr0, z0);
					glVertex3f(r * x * zr0, r * y * zr0, r * z0);
					glNormal3f(x * zr1, y * zr1, z1);
					glVertex3f(r * x * zr1, r * y * zr1, r * z1);
				}
				glEnd();
			}
		}
		void KeyFrameDisplay::removeMarking()
		{
			if (boxes.size() > 0)
			{
				boxes.clear();
				boxDim.clear();
				needRefresh = true;
				refreshPC(true, my_scaledTH, my_absTH, my_displayMode, my_minRelBS, my_sparsifyFactor);
			}
		}
		void KeyFrameDisplay::addMarking(int x1, int x2, int y1, int y2, float scaledTH, float absTH, int mode, float minBS, int sparsity)
		{
			std::vector<int> box;
			if (x1 < x2)
			{
				box.push_back(x1);
				box.push_back(x2);
			}
			else
			{
				box.push_back(x2);
				box.push_back(x1);
			}
			if (y1 < y2)
			{
				box.push_back(y1);
				box.push_back(y2);
			}
			else
			{
				box.push_back(y2);
				box.push_back(y1);
			}
			boxes.push_back(box);
			needRefresh = true;

			// printf("box %d %d %d %d %d\n", box[0], box[1], box[2], box[3], box[0] == 0);
			// printf("sizes %d %d\n", (int)boxes.size(), (int)boxDim.size());

			
			//refresh PC until the coordinates are updated
			while ( boxes.size()>boxDim.size()){
				needRefresh = true;
				refreshPC(true, scaledTH, absTH, mode, minBS, sparsity);
			}
		}
		std::vector<float> KeyFrameDisplay::getMarkingSize(int x1, int x2, int y1, int y2)
		{
			// printf("comparing to %d %d %d %d\n", x1, x2, y1, y2);
			for (int i = 0; i < boxes.size(); ++i)
			{
				std::vector<int> box = boxes[i];
				// printf("comparing %d %d %d %d\n", box[0], box[1], box[2], box[3]);
				if (box[0] == std::min(x1, x2) && box[1] == std::max(x1, x2) && box[2] == std::min(y1, y2) && box[3] == std::max(y1, y2))
				{
					// printf("boxdim size %d %d\n", (int)boxDim.size(), i);
					if (i < boxDim.size())
						return boxDim[i];
					else
						break;
				}
			}
			std::vector<float> a = {0, 0};
			return a;
		}
		int KeyFrameDisplay::getCompass(float offsetAngle)
		{

			Sophus::Matrix<float, 3, 3> m = camToWorld.rotationMatrix().cast<float>();
			// Eigen::Matrix<float, 9, 1> mat;
			// for (int i = 0; i < 3; i++)
			// {
			// 	for (int u = 0; u < 3; u++)
			// 	{
			// 		mat[u * 3 + i] = m[i * 4 + u];
			// 	}
			// }
			// printf("rotation %f %f %f\n",m(0,0),m(0,1),m(0,2));
			// printf("rotation %f %f %f\n",m(1,0),m(1,1),m(1,2));
			// printf("rotation %f %f %f\n",m(2,0),m(2,1),m(2,2));
			int yaw = -((atan2(m(2, 0), sqrt(pow(m(2, 1), 2) + pow(m(2, 2), 2))) * 180.0f / 3.14159) + offsetAngle);
			if (yaw < 0)
				yaw += 360;
			if (yaw < 0)
				yaw += 360;
			return yaw;
		}
		void KeyFrameDisplay::addPC(pcl::PointCloud<pcl::PointXYZRGB> *pcloud, float scaledTH, float absTH, int mode, float minBS, int sparsity)
		{

			// if there are no vertices, done!
			if (numSparsePoints == 0)
				return;

			my_scaledTH = scaledTH;
			my_absTH = absTH;
			my_displayMode = mode;
			my_minRelBS = minBS;
			my_sparsifyFactor = sparsity;

			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			Sophus::Vector4f tempVector;
			tempVector[3] = 1; //signifies the vector is a position in space
			// make data
			// Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints*patternNum];
			// Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints*patternNum];
			int vertexBufferNumPoints = 0;
			bool drawMarking = false;
			for (int i = 0; i < numSparsePoints; i++)
			{

				drawMarking = false;
				if (originalInputSparse[i].idpeth < 0)
					continue;
				float depth = 1.0f / originalInputSparse[i].idpeth;
				float depth4 = depth * depth;
				depth4 *= depth4;
				float var = (1.0f / (originalInputSparse[i].idepth_hessian + 0.01));

				if (var * depth4 > my_scaledTH)
					continue;

				if (var > my_absTH)
					continue;

				if (originalInputSparse[i].relObsBaseline < my_minRelBS)
					continue;

				for (auto box : boxes)
				{
					if (box[0] - wG[0] * 0.05 <= (int)originalInputSparse[i].u && box[1] + wG[0] * 0.05 >= (int)originalInputSparse[i].u && box[2] - hG[0] * 0.05 < (int)originalInputSparse[i].v && box[3] + hG[0] * 0.05 >= (int)originalInputSparse[i].v)
					{
						drawMarking = true;

						break;
					}
				}
				for (int pnt = 0; pnt < patternNum; pnt++)
				{

					if (my_sparsifyFactor > 1 && rand() % my_sparsifyFactor != 0)
						continue;
					int dx = patternP[pnt][0];
					int dy = patternP[pnt][1];
					pcl::PointXYZRGB newpoint;
					// tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
					// tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
					// tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));
					tempVector[0] = ((originalInputSparse[i].u + dx) * fxi + cxi) * depth;
					tempVector[1] = ((originalInputSparse[i].v + dy) * fyi + cyi) * depth;
					tempVector[2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f));
					tempVector = m * tempVector;
					newpoint.x = tempVector[0];
					newpoint.y = -tempVector[1];
					newpoint.z = -tempVector[2];
					newpoint.r = (uint8_t)(originalInputSparse[i].pixelcolor[pnt][2]);
					newpoint.g = (uint8_t)(originalInputSparse[i].pixelcolor[pnt][1]);
					newpoint.b = (uint8_t)(originalInputSparse[i].pixelcolor[pnt][0]);
					// tmpColorBuffer[vertexBufferNumPoints][0] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][2]);
					// tmpColorBuffer[vertexBufferNumPoints][1] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][1]);
					// tmpColorBuffer[vertexBufferNumPoints][2] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][0]);

					vertexBufferNumPoints++;

					assert(vertexBufferNumPoints <= numSparsePoints * patternNum * 2);
					pcloud->push_back(newpoint);
					if (drawMarking)
					{

						pcl::PointXYZRGB newpoint2;
						// tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
						// tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
						// tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));
						tempVector[0] = ((originalInputSparse[i].u + dx) * fxi + cxi) * depth;
						tempVector[1] = ((originalInputSparse[i].v + dy) * fyi + cyi) * depth;
						tempVector[2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f)) - 0.01;
						tempVector = m * tempVector;
						newpoint2.x = tempVector[0];
						newpoint2.y = -tempVector[1];
						newpoint2.z = -tempVector[2];
						newpoint2.r = (uint8_t)255;
						newpoint2.g = (uint8_t)0;
						newpoint2.b = (uint8_t)0;
						// tmpColorBuffer[vertexBufferNumPoints][0] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][2]);
						// tmpColorBuffer[vertexBufferNumPoints][1] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][1]);
						// tmpColorBuffer[vertexBufferNumPoints][2] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][0]);

						vertexBufferNumPoints++;

						assert(vertexBufferNumPoints <= numSparsePoints * patternNum * 2);
						pcloud->push_back(newpoint2);
					}
				}
			}
			return;
		}

		void KeyFrameDisplay::drawCam(float lineWidth, int *color, float sizeFactor)
		{
			if (width == 0)
				return;

			float sz = sizeFactor;

			glPushMatrix();

			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			glMultMatrixf((GLfloat *)m.data());

			if (color == 0)
			{
				glColor3ub(255, 0, 0);
			}
			else
				glColor3ub(color[0], color[1], color[2]);

			glLineWidth(lineWidth);
			glBegin(GL_LINES);
			glVertex3f(0, 0, 0);
			glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
			glVertex3f(0, 0, 0);
			glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

			glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
			glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

			glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
			glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

			glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
			glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

			glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
			glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

			glEnd();
			drawSphere(sz * 0.3, 10, 10);
			glPopMatrix();
		}

		void KeyFrameDisplay::drawIndexedPC(float pointSize)
		{
			// make data
			// Vec3f *tmpVertexBufferIndex = new Vec3f[numSparsePoints * patternNum];
			tmpVertexBufferIndex = new Vec3f[numSparsePoints * patternNum];
			Vec3b *tmpColorBuffer = new Vec3b[numSparsePoints * patternNum];
			int vertexBufferNumPoints = 0;
			for (int i = 0; i < numSparsePoints; i++)
			{
				if (my_displayMode == 1 && originalInputSparse[i].status != 1 && originalInputSparse[i].status != 2)
					continue;
				if (my_displayMode == 2 && originalInputSparse[i].status != 1)
					continue;
				if (my_displayMode > 2)
					continue;

				if (originalInputSparse[i].idpeth < 0)
					continue;

				float depth = 1.0f / originalInputSparse[i].idpeth;
				float depth4 = depth * depth;
				depth4 *= depth4;
				float var = (1.0f / (originalInputSparse[i].idepth_hessian + 0.01));

				if (var * depth4 > my_scaledTH)
					continue;

				if (var > my_absTH)
					continue;

				if (originalInputSparse[i].relObsBaseline < my_minRelBS)
					continue;

				for (int pnt = 0; pnt < patternNum; pnt++)
				{

					if (my_sparsifyFactor > 1 && rand() % my_sparsifyFactor != 0)
						continue;
					int dx = patternP[pnt][0];
					int dy = patternP[pnt][1];

					tmpVertexBufferIndex[vertexBufferNumPoints][0] = ((originalInputSparse[i].u + dx) * fxi + cxi) * depth;
					tmpVertexBufferIndex[vertexBufferNumPoints][1] = ((originalInputSparse[i].v + dy) * fyi + cyi) * depth;
					tmpVertexBufferIndex[vertexBufferNumPoints][2] = depth * (1 + 2 * fxi * (rand() / (float)RAND_MAX - 0.5f));
					tmpColorBuffer[vertexBufferNumPoints][0] = (unsigned char)((vertexBufferNumPoints + 1) / (256 * 256));
					tmpColorBuffer[vertexBufferNumPoints][1] = (unsigned char)(((vertexBufferNumPoints + 1) / 256) % 256);
					tmpColorBuffer[vertexBufferNumPoints][2] = (unsigned char)((vertexBufferNumPoints + 1) % 256);

					vertexBufferNumPoints++;

					assert(vertexBufferNumPoints <= numSparsePoints * patternNum);
				}
			}

			if (vertexBufferNumPoints == 0)
			{
				delete[] tmpColorBuffer;
				// delete[] tmpVertexBufferIndex;
				return;
			}

			numGLBufferGoodPoints = vertexBufferNumPoints;
			if (numGLBufferGoodPoints > numGLBufferPoints)
			{
				numGLBufferPoints = vertexBufferNumPoints * 1.3;
				vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
				colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
			}
			vertexBuffer.Upload(tmpVertexBufferIndex, sizeof(float) * 3 * numGLBufferGoodPoints, 0);
			colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char) * 3 * numGLBufferGoodPoints, 0);
			bufferValid = true;
			delete[] tmpColorBuffer;
			// delete[] tmpVertexBufferIndex;

			drawPC(pointSize, 0);
			numPoints = vertexBufferNumPoints;
		}
		Vec3f KeyFrameDisplay::getPCfromID(int id)
		{
			if (id > numPoints)
			{
				Vec3f result;
				result[0] = 0;
				result[1] = 0;
				result[2] = 0;
				return result;
			}
			Vec3f result;
			for (int i = 0; i < 3; i++)
				result[i] = tmpVertexBufferIndex[id][i];

			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			Sophus::Vector4f tempVector;
			tempVector[3] = 1; //signifies the vector is a position in space
			for (int i = 0; i < 3; i++)
				tempVector[i] = result[i];
			tempVector = m * tempVector;
			for (int i = 0; i < 3; i++)
				result[i] = tempVector[i];
			delete[] tmpVertexBufferIndex;
			return result;
		}

		Vec3f KeyFrameDisplay::getPCbyMatrix(GLdouble cursor_pos[3])
		{

			Vec3f result;
			int best_ind = -1;
			int best_score = 9999999;
			int score = 0;
			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			Sophus::Vector4f tempVector;
			tempVector[3] = 1; //signifies the vector is a position in space
			for (int ind = 0; ind < numPoints; ind++)
			{
				for (int i = 0; i < 3; i++)
					result[i] = tmpVertexBufferIndex[id][i];
				//flip y axis
				result[1] = -result[1];
				for (int i = 0; i < 3; i++)
					tempVector[i] = result[i];
				tempVector = m * tempVector;
				for (int i = 0; i < 3; i++)
					result[i] = tempVector[i];
				score = abs(tempVector[0] - cursor_pos[0]) + abs(tempVector[1] - cursor_pos[1]) + abs(tempVector[2] - cursor_pos[2]);
				if (score < best_score)
				{
					best_ind = ind;
					best_score = score;
				}
			}

			if (best_ind == -1)
			{
				result[0] = 0;
				result[1] = 0;
				result[2] = 0;
				return result;
			}
			for (int i = 0; i < 3; i++)
				result[i] = tmpVertexBufferIndex[best_ind][i];

			m = camToWorld.matrix().cast<float>();
			// Sophus::Vector4f tempVector;
			tempVector[3] = 1; //signifies the vector is a position in space
			for (int i = 0; i < 3; i++)
				tempVector[i] = result[i];
			tempVector = m * tempVector;
			for (int i = 0; i < 3; i++)
				result[i] = tempVector[i];

			printf("result %f %f %f", tempVector[0], tempVector[1], tempVector[2]);
			return result;
		}
		void KeyFrameDisplay::drawPC(float pointSize, int *color)
		{

			if (!bufferValid || numGLBufferGoodPoints == 0)
				return;

			glDisable(GL_LIGHTING);

			glPushMatrix();

			Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
			glMultMatrixf((GLfloat *)m.data());

			glPointSize(pointSize);

			if (color == 0)
			{
				// glColor3ub(255,255,255);
				colorBuffer.Bind();
				glColorPointer(colorBuffer.count_per_element, colorBuffer.datatype, 0, 0);
				glEnableClientState(GL_COLOR_ARRAY);
			}
			else
			{
				glColor3ub(color[0], color[1], color[2]);
			}

			vertexBuffer.Bind();
			glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);
			glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
			glDisableClientState(GL_VERTEX_ARRAY);
			vertexBuffer.Unbind();

			if (color == 0)
			{
				glDisableClientState(GL_COLOR_ARRAY);
				colorBuffer.Unbind();
			}
			glPopMatrix();
		}

	}
}
