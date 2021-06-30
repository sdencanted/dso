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
	originalInputSparse = 0;
	numSparseBufferSize=0;
	numSparsePoints=0;

	id = 0;
	active= true;
	camToWorld = SE3();

	needRefresh=true;

	my_scaledTH =1e10;
	my_absTH = 1e10;
	my_displayMode = 1;
	my_minRelBS = 0;
	my_sparsifyFactor = 1;

	numGLBufferPoints=0;
	bufferValid = false;
}
void KeyFrameDisplay::setFromF(FrameShell* frame, CalibHessian* HCalib)
{
	id = frame->id;
	fx = HCalib->fxl();
	fy = HCalib->fyl();
	cx = HCalib->cxl();
	cy = HCalib->cyl();
	width = wG[0];
	height = hG[0];
	fxi = 1/fx;
	fyi = 1/fy;
	cxi = -cx / fx;
	cyi = -cy / fy;
	camToWorld = frame->camToWorld;
	needRefresh=true;
}

void KeyFrameDisplay::setFromKF(FrameHessian* fh, CalibHessian* HCalib)
{
	setFromF(fh->shell, HCalib);

	// add all traces, inlier and outlier points.
	int npoints = 	fh->immaturePoints.size() +
					fh->pointHessians.size() +
					fh->pointHessiansMarginalized.size() +
					fh->pointHessiansOut.size();

	if(numSparseBufferSize < npoints)
	{
		if(originalInputSparse != 0) delete originalInputSparse;
		numSparseBufferSize = npoints+100;
        originalInputSparse = new InputPointSparse<MAX_RES_PER_POINT>[numSparseBufferSize];
	}

    InputPointSparse<MAX_RES_PER_POINT>* pc = originalInputSparse;
	numSparsePoints=0;
	for(ImmaturePoint* p : fh->immaturePoints)
	{
		for(int i=0;i<patternNum;i++){
			pc[numSparsePoints].color[i] = p->color[i]; // unused
			for(int u=0;u<3;u++)
				pc[numSparsePoints].pixelcolor[i][u] = p->pixelcolor[i][u];}

		pc[numSparsePoints].u = p->u;
		pc[numSparsePoints].v = p->v;	
		pc[numSparsePoints].idpeth = (p->idepth_max+p->idepth_min)*0.5f;
		pc[numSparsePoints].idepth_hessian = 1000;
		pc[numSparsePoints].relObsBaseline = 0;
		pc[numSparsePoints].numGoodRes = 1;
		pc[numSparsePoints].status = 0;
		numSparsePoints++;
	}

	for(PointHessian* p : fh->pointHessians)
	{
		for(int i=0;i<patternNum;i++){
			pc[numSparsePoints].color[i] = p->color[i]; // unused
			for(int u=0;u<3;u++)
				pc[numSparsePoints].pixelcolor[i][u] = p->pixelcolor[i][u];}
		pc[numSparsePoints].u = p->u;
		pc[numSparsePoints].v = p->v;
		pc[numSparsePoints].idpeth = p->idepth_scaled;
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=1;

		numSparsePoints++;
	}

	for(PointHessian* p : fh->pointHessiansMarginalized)
	{
		for(int i=0;i<patternNum;i++){
			pc[numSparsePoints].color[i] = p->color[i];
			for(int u=0;u<3;u++)
				pc[numSparsePoints].pixelcolor[i][u] = p->pixelcolor[i][u];}
		pc[numSparsePoints].u = p->u;
		pc[numSparsePoints].v = p->v;
		pc[numSparsePoints].idpeth = p->idepth_scaled;
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=2;
		numSparsePoints++;
	}

	for(PointHessian* p : fh->pointHessiansOut)
	{
		for(int i=0;i<patternNum;i++){
			pc[numSparsePoints].color[i] = p->color[i];// unused
			for(int u=0;u<3;u++)
				pc[numSparsePoints].pixelcolor[i][u] = p->pixelcolor[i][u];}
		pc[numSparsePoints].u = p->u;
		pc[numSparsePoints].v = p->v;
		pc[numSparsePoints].idpeth = p->idepth_scaled;
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=3;
		numSparsePoints++;
	}
	assert(numSparsePoints <= npoints);

	camToWorld = fh->PRE_camToWorld;
	needRefresh=true;
}


KeyFrameDisplay::~KeyFrameDisplay()
{
	if(originalInputSparse != 0)
		delete[] originalInputSparse;
}

bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity)
{
	if(canRefresh)
	{
		needRefresh = needRefresh ||
				my_scaledTH != scaledTH ||
				my_absTH != absTH ||
				my_displayMode != mode ||
				my_minRelBS != minBS ||
				my_sparsifyFactor != sparsity;
	}

	if(!needRefresh) return false;
	needRefresh=false;

	my_scaledTH = scaledTH;
	my_absTH = absTH;
	my_displayMode = mode;
	my_minRelBS = minBS;
	my_sparsifyFactor = sparsity;


	// if there are no vertices, done!
	if(numSparsePoints == 0)
		return false;

	// make data
	Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints*patternNum*2];
	Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints*patternNum*2];
	int vertexBufferNumPoints=0;
	bool drawMarking=false;
	for(int i=0;i<numSparsePoints;i++)
	{
		/* display modes:
		 * my_displayMode==0 - all pts, color-coded
		 * my_displayMode==1 - normal points
		 * my_displayMode==2 - active only
		 * my_displayMode==3 - nothing
		 */			
		
		drawMarking=false;
		for(auto box: boxes){
			if (box[0]-wG[0]*0.05<=(int)originalInputSparse[i].u && box[1]+wG[0]*0.05>=(int)originalInputSparse[i].u && box[2]-hG[0]*0.05<(int)originalInputSparse[i].v &&box[3]+hG[0]*0.05>=(int)originalInputSparse[i].v ){
				drawMarking=true;
				printf("a\n\n\n\n\n\n\n\na");
				break;
			}
		}
		if(my_displayMode==1 && originalInputSparse[i].status != 1 && originalInputSparse[i].status!= 2) continue;
		if(my_displayMode==2 && originalInputSparse[i].status != 1) continue;
		if(my_displayMode>2) continue;

		if(originalInputSparse[i].idpeth < 0) continue;


		float depth = 1.0f / originalInputSparse[i].idpeth;
		float depth4 = depth*depth; depth4*= depth4;
		float var = (1.0f / (originalInputSparse[i].idepth_hessian+0.01));

		if(var * depth4 > my_scaledTH)
			continue;

		if(var > my_absTH)
			continue;

		if(originalInputSparse[i].relObsBaseline < my_minRelBS)
			continue;


		for(int pnt=0;pnt<patternNum;pnt++)
		{

			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;
			int dx = patternP[pnt][0];
			int dy = patternP[pnt][1];

			tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));
			tmpColorBuffer[vertexBufferNumPoints][0] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][2]);
			tmpColorBuffer[vertexBufferNumPoints][1] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][1]);
			tmpColorBuffer[vertexBufferNumPoints][2] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][0]);
			if(drawMarking){
				tmpVertexBuffer[vertexBufferNumPoints+1][0] = ((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
				tmpVertexBuffer[vertexBufferNumPoints+1][1] = ((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
				tmpVertexBuffer[vertexBufferNumPoints+1][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f)) - 0.01;

				tmpColorBuffer[vertexBufferNumPoints+1][0] = (unsigned char)255;
				tmpColorBuffer[vertexBufferNumPoints+1][1] = (unsigned char)0;
				tmpColorBuffer[vertexBufferNumPoints+1][2] = (unsigned char)0;
			}

			
			vertexBufferNumPoints++;
			if(drawMarking)
				vertexBufferNumPoints++;

			assert(vertexBufferNumPoints <= numSparsePoints*patternNum*2);
		}
	}

	if(vertexBufferNumPoints==0)
	{
		delete[] tmpColorBuffer;
		delete[] tmpVertexBuffer;
		return true;
	}

	numGLBufferGoodPoints = vertexBufferNumPoints;
	if(numGLBufferGoodPoints > numGLBufferPoints)
	{
		numGLBufferPoints = vertexBufferNumPoints*1.3;
		vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
		colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
	}
	vertexBuffer.Upload(tmpVertexBuffer, sizeof(float)*3*numGLBufferGoodPoints, 0);
	colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char)*3*numGLBufferGoodPoints, 0);
	bufferValid=true;
	delete[] tmpColorBuffer;
	delete[] tmpVertexBuffer;


	return true;
}

void drawSphere(double r, int lats, int longs) {
    int i, j;
    for(i = 0; i <= lats; i++) {
        double lat0 = M_PI * (-0.5 + (double) (i - 1) / lats);
        double z0  = sin(lat0);
        double zr0 =  cos(lat0);

        double lat1 = M_PI * (-0.5 + (double) i / lats);
        double z1 = sin(lat1);
        double zr1 = cos(lat1);

        glBegin(GL_QUAD_STRIP);
        for(j = 0; j <= longs; j++) {
            double lng = 2 * M_PI * (double) (j - 1) / longs;
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
void KeyFrameDisplay::removeMarking(){
	if (boxes.size()>0){
		boxes.clear();
		needRefresh=true;
		refreshPC(true, my_scaledTH,my_absTH, my_displayMode, my_minRelBS, my_sparsifyFactor);
	}
}
void KeyFrameDisplay::addMarking(int x1,int x2, int y1, int y2){
	std::vector<int> box;
	if(x1<x2){
		box.push_back(x1);
		box.push_back(x2);
	}
	else{
		box.push_back(x2);
		box.push_back(x1);
	}
	if(y1<y2){
		box.push_back(y1);
		box.push_back(y2);
	}
	else{
		box.push_back(y2);
		box.push_back(y1);
	}
	boxes.push_back(box);
	needRefresh=true;
	refreshPC(true, my_scaledTH,my_absTH, my_displayMode, my_minRelBS, my_sparsifyFactor);

}
void KeyFrameDisplay::addPC(pcl::PointCloud<pcl::PointXYZRGB>* pcloud,float scaledTH, float absTH, int mode, float minBS, int sparsity){

	// if there are no vertices, done!
	if(numSparsePoints == 0)
		return;

	my_scaledTH = scaledTH;
	my_absTH = absTH;
	my_displayMode = mode;
	my_minRelBS = minBS;
	my_sparsifyFactor = sparsity;

	Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
	Sophus::Vector4f tempVector;
	tempVector[3]=1;//signifies the vector is a position in space
	// make data
	// Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints*patternNum];
	// Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints*patternNum];
	int vertexBufferNumPoints=0;
	bool drawMarking=false;
	for(int i=0;i<numSparsePoints;i++)
	{

		drawMarking=false;
		for(auto box: boxes){
			if (box[0]-wG[0]*0.05<=(int)originalInputSparse[i].u && box[1]+wG[0]*0.05>=(int)originalInputSparse[i].u && box[2]-hG[0]*0.05<(int)originalInputSparse[i].v &&box[3]+hG[0]*0.05>=(int)originalInputSparse[i].v ){
				drawMarking=true;
				// printf("a\n\n\n\n\n\n\n\na");
				break;
			}
		}
		if(originalInputSparse[i].idpeth < 0) continue;
		float depth = 1.0f / originalInputSparse[i].idpeth;
		float depth4 = depth*depth; depth4*= depth4;
		float var = (1.0f / (originalInputSparse[i].idepth_hessian+0.01));

		if(var * depth4 > my_scaledTH)
			continue;

		if(var > my_absTH)
			continue;

		if(originalInputSparse[i].relObsBaseline < my_minRelBS)
			continue;


		for(int pnt=0;pnt<patternNum;pnt++)
		{
			
			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;
			int dx = patternP[pnt][0];
			int dy = patternP[pnt][1];
			pcl::PointXYZRGB newpoint;
			// tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
			// tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
			// tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));
			tempVector[0]=((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
			tempVector[1]=((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
			tempVector[2]=depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));
			tempVector=m*tempVector;
			newpoint.x=tempVector[0];
			newpoint.y=-tempVector[1];
			newpoint.z=-tempVector[2];
			newpoint.r=(uint8_t)(originalInputSparse[i].pixelcolor[pnt][2]);
			newpoint.g=(uint8_t)(originalInputSparse[i].pixelcolor[pnt][1]);
			newpoint.b=(uint8_t)(originalInputSparse[i].pixelcolor[pnt][0]);
			// tmpColorBuffer[vertexBufferNumPoints][0] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][2]);
			// tmpColorBuffer[vertexBufferNumPoints][1] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][1]);
			// tmpColorBuffer[vertexBufferNumPoints][2] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][0]);
			
			vertexBufferNumPoints++;


			assert(vertexBufferNumPoints <= numSparsePoints*patternNum*2);
			pcloud->push_back(newpoint);
			if(drawMarking){

				pcl::PointXYZRGB newpoint2;
				// tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
				// tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
				// tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));
				tempVector[0]=((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
				tempVector[1]=((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
				tempVector[2]=depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f))-0.01;
				tempVector=m*tempVector;
				newpoint2.x=tempVector[0];
				newpoint2.y=-tempVector[1];
				newpoint2.z=-tempVector[2];
				newpoint2.r=(uint8_t)255;
				newpoint2.g=(uint8_t)0;
				newpoint2.b=(uint8_t)0;
				// tmpColorBuffer[vertexBufferNumPoints][0] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][2]);
				// tmpColorBuffer[vertexBufferNumPoints][1] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][1]);
				// tmpColorBuffer[vertexBufferNumPoints][2] = (unsigned char)(originalInputSparse[i].pixelcolor[pnt][0]);
				
				vertexBufferNumPoints++;


				assert(vertexBufferNumPoints <= numSparsePoints*patternNum*2);
				pcloud->push_back(newpoint2);
			}
		}
	}

	return;
}

void KeyFrameDisplay::drawCam(float lineWidth, int* color, float sizeFactor)
{
	if(width == 0)
		return;

	float sz=sizeFactor;

	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
		glMultMatrixf((GLfloat*)m.data());

		if(color == 0)
		{
			glColor3ub(255,0,0);
		}
		else
			glColor3ub(color[0],color[1],color[2]);

		glLineWidth(lineWidth);
		glBegin(GL_LINES);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);

		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);

		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);

		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);

		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);

		glEnd();
		drawSphere(sz*0.3,5,5);
	glPopMatrix();
}


void KeyFrameDisplay::drawPC(float pointSize,int* color)
{

	if(!bufferValid || numGLBufferGoodPoints==0)
		return;


	glDisable(GL_LIGHTING);

	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
		glMultMatrixf((GLfloat*)m.data());

		glPointSize(pointSize);


		colorBuffer.Bind();
		if(color == 0 || true)
		{
			// glColor3ub(255,255,255);
			glColorPointer(colorBuffer.count_per_element, colorBuffer.datatype, 0, 0);
		}
		else{
			glColor3ub(color[0],color[1],color[2]);
		}
		glEnableClientState(GL_COLOR_ARRAY);

		vertexBuffer.Bind();
		glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
		glDisableClientState(GL_VERTEX_ARRAY);
		vertexBuffer.Unbind();

		glDisableClientState(GL_COLOR_ARRAY);
		colorBuffer.Unbind();

	glPopMatrix();
}

}
}
