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

#include <thread>
#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"
#include <boost/thread.hpp>
#include <boost/format.hpp>

#include "util/settings.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"
#include <pangolin/gl/glfont.h>
#include <pangolin/gl/gltext.h>
#include <pangolin/display/display_internal.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/pixel_format.h>
#include <pangolin/image/typed_image.h>

// to detect mouse click and select object
namespace pangolin
{
	struct MyHandler3D : Handler3D
	{
	protected:
		bool last_mousedown = false;
		bool first_mousedown = false;
		bool *checkObject;
		int *rx;
		int *ry;
		int mousedownx = 0;
		int mousedowny = 0;

	public:
		MyHandler3D(OpenGlRenderState &cam_state, bool &checkObject, int &rx, int &ry,
					AxisDirection enforce_up = AxisNone,
					float trans_scale = 0.01f,
					float zoom_fraction = PANGO_DFLT_HANDLER3D_ZF)
			: Handler3D(cam_state, enforce_up, trans_scale, zoom_fraction), checkObject(&checkObject), rx(&rx), ry(&ry){};
		void Mouse(View &display,
				   MouseButton button,
				   int x,
				   int y,
				   bool pressed,
				   int button_state)
		{
			// mouse down
			last_pos[0] = (float)x;
			last_pos[1] = (float)y;

			GLprecision T_nc[3 * 4];
			LieSetIdentity(T_nc);

			funcKeyState = 0;

			if (pressed)
			{
				first_mousedown = !last_mousedown;
				if (first_mousedown && (button == MouseButtonLeft))
				{
					// printf("left click\n");
					mousedownx = x;
					mousedowny = y;
				}
				GetPosNormal(display, x, y, p, Pw, Pc, n, last_z);
				if (ValidWinDepth(p[2]))
				{
					last_z = p[2];
					std::copy(Pc, Pc + 3, rot_center);
				}

				if (button == MouseWheelUp || button == MouseWheelDown)
				{
					LieSetIdentity(T_nc);
					const GLprecision t[3] = {0, 0, (button == MouseWheelUp ? 1 : -1) * 100 * tf};
					LieSetTranslation<>(T_nc, t);
					if (!(button_state & MouseButtonRight) && !(rot_center[0] == 0 && rot_center[1] == 0 && rot_center[2] == 0))
					{
						LieSetTranslation<>(T_nc, rot_center);
						const GLprecision s = (button == MouseWheelUp ? -1.0 : 1.0) * zf;
						MatMul<3, 1>(T_nc + (3 * 3), s);
					}
					OpenGlMatrix &spec = cam_state->GetModelViewMatrix();
					LieMul4x4bySE3<>(spec.m, T_nc, spec.m);
				}

				funcKeyState = button_state;
				last_mousedown = true;
			}
			else if (button_state == 0 && (button == MouseButtonLeft))
			{
				last_mousedown = false;
				// printf("left click released x %d y %d mx %d my %d\n", x, y, mousedownx, mousedowny);
				if (x == mousedownx && y == mousedowny)
				{
					*rx = x;
					*ry = y;
					*checkObject = true;
				}
			}
		}
	};
	// to draw boxes in display
	struct MyHandler2D : Handler
	{
	protected:
		bool last_mousedown = false;
		bool *checkfirst;
		int *firstx;
		int *firsty;
		bool *checksecond;
		bool *releasesecond;
		int *secondx;
		int *secondy;

	public:
		MyHandler2D(bool &checkfirst, int &firstx, int &firsty, bool &checksecond, int &secondx, int &secondy, bool &releasesecond) : Handler(), checkfirst(&checkfirst), firstx(&firstx), firsty(&firsty), checksecond(&checksecond), secondx(&secondx), secondy(&secondy), releasesecond(&releasesecond){};
		void Mouse(View &d, MouseButton button, int x, int y, bool pressed, int button_state)
		{
			if (pressed)
			{
				if (!last_mousedown && (button == MouseButtonLeft))
				{
					// printf("left click down\n");
					*firstx = x;
					*firsty = y;
					*checkfirst = true;
					*secondx = x;
					*secondy = y;
					*checksecond = true;
				}
				last_mousedown = true;
			}
			else if (button_state == 0 && (button == MouseButtonLeft))
			{
				last_mousedown = false;
				*releasesecond = true;
				// printf("left click released x %d y %d\n", x, y);
			}
		}
		void MouseMotion(View &, int x, int y, int button_state)
		{
			if (button_state)
			{
				if (*secondx != x && *secondy != y)
				{
					// printf("left click moving\n");
					*secondx = x;
					*secondy = y;
					*checksecond = true;
				}
			}
		}
	};
}

namespace pangolin
{
	//for drawing text
	extern "C" const unsigned char AnonymousPro_ttf[];
}
namespace dso
{
	namespace IOWrap
	{
		// to hold information on marking boxes made, as well as screenshots of the scene and point cloud
		struct framedata
		{
			std::vector<std::vector<float>> markings;
			pangolin::TypedImage pointcloud;
			pangolin::TypedImage image;
			double timestamp;
			framedata() : markings(), pointcloud(), image(), timestamp() {}
		};
		PangolinDSOViewer::PangolinDSOViewer(int w, int h, bool startRunThread)
		{
			this->w = w;
			this->h = h;
			running = true;
			bool selectedkfchange = false;
			int selectedkf = -1;
			{
				boost::unique_lock<boost::mutex> lk(openImagesMutex);
				internalVideoImg = new MinimalImageB3(w, h);
				internalVideoPlayerImg = new MinimalImageB3(w, h);
				internalKFImg = new MinimalImageB3(w, h);
				videoImgChanged = videoPlayerImgChanged = kfImgChanged = resImgChanged = true;

				internalVideoImg->setBlack();
				internalVideoPlayerImg->setBlack();
				internalKFImg->setBlack();
				// internalResImg->setBlack();
			}

			{
				currentCam = new KeyFrameDisplay();
			}

			needReset = false;

			if (startRunThread)
				runThread = boost::thread(&PangolinDSOViewer::run, this);
		}
		pangolin::TypedImage HoldFramebuffer(const pangolin::Viewport &v)
		{
			pangolin::PixelFormat fmt = pangolin::PixelFormatFromString("RGBA32");
			pangolin::TypedImage buffer(v.w, v.h, fmt);
			glReadBuffer(GL_BACK);
			glPixelStorei(GL_PACK_ALIGNMENT, 1); // TODO: Avoid this?
			glReadPixels(v.l, v.b, v.w, v.h, GL_RGBA, GL_UNSIGNED_BYTE, buffer.ptr);
			// SaveImage(buffer, fmt, prefix + ".png", false);
			return buffer;
		}
		int PangolinDSOViewer::getselectedkf()
		{

			if (selectedkfchange)
			{
				selectedkfchange = false;
				return selectedkf;
			}
			else
				return -1;
		}
		PangolinDSOViewer::~PangolinDSOViewer()
		{
			close();
			runThread.join();
		}

		void PangolinDSOViewer::run()
		{
			printf("START PANGOLIN!\n");
			int returnId;
			int blue[3] = {0, 0, 255};
			int green[3] = {0, 255, 0};
			int yellow[3] = {255, 255, 0};
			selectedkf = -1;
			int selectedkf_index = -1;
			bool updated_frame = true;
			int firstx = 1000;
			int firsty = 1000;
			int secondx = 1000;
			int secondy = 1000;
			bool checkfirst = false;
			bool checksecond = false;
			bool releasesecond = false;
			float firsthorizontal = 0;
			float firstvertical = 0;
			float secondhorizontal = 0;
			float secondvertical = 0;
			int playback_mode = PAUSE;
			double frameseconds = clock() / (float)(CLOCKS_PER_SEC);
			std::map<int, framedata> markings;

			pangolin::GlFont mybigfont(pangolin::AnonymousPro_ttf, 30);
			pangolin::GlFont myfont(pangolin::AnonymousPro_ttf, 15);
			pangolin::CreateWindowAndBind("Main", 2 * w, 2 * h);
			GLubyte pixel[4]; // used to detect the object clicked by the user
			glEnable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
			const int UI_WIDTH = 180;

			// 3D visualization
			pangolin::OpenGlRenderState Visualization3D_camera(
				pangolin::ProjectionMatrix(2 * w / 5, h, 400, 400, 2 * w / 10, h / 2, 0.1, 1000),
				pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY));
			bool checkObject = false;
			int rx = 0;
			int ry = 0;
			pangolin::View &Visualization3D_display = pangolin::CreateDisplay()
														  .SetBounds(0, 1.0, pangolin::Attach::Pix(UI_WIDTH + 1), 0.5, -(2 * w / 5) / (float)h)
														  .SetHandler(new pangolin::MyHandler3D(Visualization3D_camera, checkObject, rx, ry));

			// 3 images + player
			pangolin::View &d_kfDepth = pangolin::Display("imgKFDepth")
											.SetAspect(w / (float)h);

			pangolin::View &d_video = pangolin::Display("imgVideo")
										  .SetAspect(w / (float)h);

			// pangolin::View &d_residual = pangolin::Display("imgResidual")
			// 								 .SetAspect(w / (float)h);

			pangolin::View &d_video_player = pangolin::Display("imgVideoPlayer")
												 .SetAspect(w / (float)h)
												 .SetHandler(new pangolin::MyHandler2D(checkfirst, firstx, firsty, checksecond, secondx, secondy, releasesecond));
			pangolin::View &d_video_player_text = pangolin::Display("imgVideoPlayer")
													  .SetAspect(w / (float)h)
													  .SetHandler(new pangolin::MyHandler2D(checkfirst, firstx, firsty, checksecond, secondx, secondy, releasesecond));

			pangolin::GlTexture texKFDepth(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
			pangolin::GlTexture texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
			pangolin::GlTexture texVideoPlayer(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
			// pangolin::GlTexture texResidual(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
			pangolin::CreateDisplay()
				.SetBounds(0.0, 0.54, 0.5, 1)
				.AddDisplay(d_video_player);
			pangolin::CreateDisplay()
				.SetBounds(0.0, 0.54, 0.5, 1)
				.AddDisplay(d_video_player_text);
			pangolin::CreateDisplay()
				.SetBounds(0.54, 1, 0.5, 1)
				.SetLayout(pangolin::LayoutEqual)
				.AddDisplay(d_kfDepth)
				.AddDisplay(d_video);
			//    .AddDisplay(d_residual);

			// parameter reconfigure gui
			pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

			pangolin::Var<int> settings_pointCloudMode("ui.PC_mode", 1, 1, 4, false);

			pangolin::Var<bool> settings_showKFCameras("ui.KFCam", true, true);
			pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam", true, true);
			pangolin::Var<bool> settings_showTrajectory("ui.Trajectory", true, true);
			pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory", false, true);
			pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst", true, true);
			pangolin::Var<bool> settings_showAllConstraints("ui.AllConst", false, true);

			pangolin::Var<bool> settings_show3D("ui.show3D", true, true);
			pangolin::Var<bool> settings_showLiveDepth("ui.showDepth", true, true);
			pangolin::Var<bool> settings_showLiveVideo("ui.showVideo", true, true);
			pangolin::Var<bool> settings_showLiveResidual("ui.showResidual", false, true);

			pangolin::Var<bool> settings_showFramesWindow("ui.showFramesWindow", false, true);
			pangolin::Var<bool> settings_showFullTracking("ui.showFullTracking", false, true);
			pangolin::Var<bool> settings_showCoarseTracking("ui.showCoarseTracking", false, true);

			pangolin::Var<int> settings_sparsity("ui.sparsity", 1, 1, 20, false);
			pangolin::Var<double> settings_scaledVarTH("ui.relVarTH", 0.001, 1e-10, 1e10, true);
			pangolin::Var<double> settings_absVarTH("ui.absVarTH", 0.001, 1e-10, 1e10, true);
			pangolin::Var<double> settings_minRelBS("ui.minRelativeBS", 0.1, 0, 1, false);

			pangolin::Var<bool> settings_resetButton("ui.Reset", false, false);

			pangolin::Var<int> settings_nPts("ui.activePoints", setting_desiredPointDensity, 50, 5000, false);
			pangolin::Var<int> settings_nCandidates("ui.pointCandidates", setting_desiredImmatureDensity, 50, 5000, false);
			pangolin::Var<int> settings_nMaxFrames("ui.maxFrames", setting_maxFrames, 4, 10, false);
			pangolin::Var<double> settings_kfFrequency("ui.kfFrequency", setting_kfGlobalWeight, 0.1, 3, false);
			pangolin::Var<double> settings_gradHistAdd("ui.minGradAdd", setting_minGradHistAdd, 0, 15, false);

			pangolin::Var<double> settings_trackFps("ui.Track fps", 0, 0, 0, false);
			pangolin::Var<double> settings_mapFps("ui.KF fps", 0, 0, 0, false);
			pangolin::Var<double> settings_playbackFps("ui.Playback FPS", setting_kfGlobalWeight, 0.1, 30, false);
			pangolin::Var<bool> settings_playbackForwardButton("ui.Forward", false, false);
			pangolin::Var<bool> settings_playbackReverseButton("ui.Reverse", false, false);
			pangolin::Var<bool> settings_playbackPauseButton("ui.Pause", false, false);
			pangolin::Var<bool> settings_deleteAllMarkings("ui.Delete All Markings", false, false);
			pangolin::Var<bool> settings_deleteMarkings("ui.Delete Frame Markings", false, false);
			pangolin::Var<bool> settings_saveMarkings("ui.Save Markings", false, false);
			std::string marking_text = "eg. fault";
			bool saveimage = false;

			// Default hooks for exiting (Esc) and fullscreen (tab).
			while (!pangolin::ShouldQuit() && running)
			{
				// Clear entire screen
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				if (setting_render_display3D)
				{

					if (checkObject)
					{
						checkObject = false;
						printf("received\n");
						glDrawBuffer(GL_BACK);

						// Activate efficiently by object
						Visualization3D_display.Activate(Visualization3D_camera);
						boost::unique_lock<boost::mutex> lk3d(model3DMutex);

						for (KeyFrameDisplay *fh : keyframes)
						{
							int fhr = (int)((uint)(fh->id + 1) >> 16);
							int fhg = (int)(((uint)(fh->id + 1) << 16) >> 24);
							int fhb = (int)(((uint)(fh->id + 1) << 24) >> 24);
							int kfcolor[3] = {fhr, fhg, fhb};
							fh->drawCam(1, kfcolor, 0.1);
						}
						int kfcolor[3] = {255, 255, 255};
						currentCam->drawCam(2, kfcolor, 0.2);
						// drawConstraints();

						glReadBuffer(GL_BACK);
						glReadPixels(rx, ry, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
						printf("R: %d	 G: %d	 B: %d\n", pixel[0], pixel[1], pixel[2]);
						returnId = (int)(pixel[0] << 16) + (int)(pixel[1] << 8) + (int)(pixel[2]) - 1;

						if (returnId == -1)
						{ //nothing
							printf("clicked on nothing!\n");
							if (selectedkf != -1)
							{
								selectedkf = -1;
								selectedkfchange = true;
							}
						}
						else if (returnId == 16777214)
						{ //current frame
							printf("clicked on current frame!\n");
							if (selectedkf != -1)
							{
								selectedkf = -1;
								selectedkfchange = true;
							}
						}
						else
						{ //clicked some other frame perhaps
							if (selectedkf != returnId)
							{

								for (KeyFrameDisplay *fh : keyframes)
								{
									if (returnId == fh->id)
									{
										selectedkf = returnId;
										selectedkfchange = true;
										for (int i = 0; i < keyframes.size(); ++i)
										{
											if (keyframes[i]->id == selectedkf)
											{
												selectedkf_index = i;
												break;
											}
										}
										firsthorizontal = 0;
										secondhorizontal = 0;
										firstvertical = 0;
										secondvertical = 0;
										break;
									}
								}
							}
							printf("clicked on ID %d with array position %d\n", returnId, selectedkf_index);
						}
						lk3d.unlock();
						glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
					}
					// Activate efficiently by object
					Visualization3D_display.Activate(Visualization3D_camera);
					boost::unique_lock<boost::mutex> lk3d(model3DMutex);
					//pangolin::glDrawColouredCube();
					int refreshed = 0;
					for (KeyFrameDisplay *fh : keyframes)
					{
						if (this->settings_showKFCameras)
							if (selectedkf == fh->id && selectedkf != -1)
							{

								fh->drawCam(1, blue, 0.16);
							}
							else if (markings.find(fh->id) != markings.end())
							{
								fh->drawCam(1, yellow, 0.1);
							}
							else
							{
								fh->drawCam(1, green, 0.1);
							}

						refreshed += (int)(fh->refreshPC(refreshed < 10, this->settings_scaledVarTH, this->settings_absVarTH,
														 this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity));

						if (this->settings_showKFCameras)
						{
							if (selectedkf == fh->id && selectedkf != -1)
							{

								fh->drawPC(1, blue);
							}
							else
							{
								fh->drawPC(1, 0);
							}
						}
					}
					if (this->settings_showCurrentCamera)
					{
						currentCam->drawCam(2, 0, 0.2);
					}
					drawConstraints();
					lk3d.unlock();
				}
				glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
				openImagesMutex.lock();
				if (videoImgChanged)
					texVideo.Upload(internalVideoImg->data, GL_BGR, GL_UNSIGNED_BYTE);
				if (kfImgChanged)
					texKFDepth.Upload(internalKFImg->data, GL_BGR, GL_UNSIGNED_BYTE);
				// if (resImgChanged)
				// 	texResidual.Upload(internalResImg->data, GL_BGR, GL_UNSIGNED_BYTE);
				if (videoPlayerImgChanged)
				{
					texVideoPlayer.Upload(internalVideoPlayerImg->data, GL_BGR, GL_UNSIGNED_BYTE);
					updated_frame = true;
				}

				videoImgChanged = kfImgChanged = resImgChanged = false;
				videoPlayerImgChanged = false;
				openImagesMutex.unlock();

				// update fps counters
				{
					openImagesMutex.lock();
					float sd = 0;
					for (float d : lastNMappingMs)
						sd += d;
					settings_mapFps = lastNMappingMs.size() * 1000.0f / sd;
					openImagesMutex.unlock();
				}
				{
					model3DMutex.lock();
					float sd = 0;
					for (float d : lastNTrackingMs)
						sd += d;
					settings_trackFps = lastNTrackingMs.size() * 1000.0f / sd;
					model3DMutex.unlock();
				}

				if (setting_render_displayVideo)
				{
					if (selectedkf != -1)
					{
						d_video.Activate();
						glColor3ub(255, 0, 0);
						myfont.Text("original video").DrawWindow(d_video.GetBounds().l, d_video.GetBounds().b - 1.0f * myfont.Height());
						glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
						texVideo.RenderToViewportFlipY();
					}
					d_video_player.Activate();
					glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
					if (checkfirst)
					{
						checkfirst = false;
						firsthorizontal = std::max(-1.0, std::min(1.0, (float)(firstx - d_video_player.GetBounds().l) / (float)(0.5 * d_video_player.GetBounds().w) - 1.0));
						firstvertical = std::max(-1.0, std::min(1.0, (float)((firsty)-d_video_player.GetBounds().b) / (float)(0.5 * d_video_player.GetBounds().h) - 1.0));
						printf("%f %f", firsthorizontal, firstvertical);
					}
					if (checksecond)
					{
						checksecond = false;

						secondhorizontal = std::max(-1.0, std::min(1.0, (float)(secondx - d_video_player.GetBounds().l) / (float)(0.5 * d_video_player.GetBounds().w) - 1.0));
						secondvertical = std::max(-1.0, std::min(1.0, (float)((secondy)-d_video_player.GetBounds().b) / (float)(0.5 * d_video_player.GetBounds().h) - 1.0));
					}
					if (releasesecond)
					{
						releasesecond = false;
						if (selectedkf != -1)
						{
							saveimage = true;
							std::vector<float> coords;
							coords.push_back(firsthorizontal);
							coords.push_back(firstvertical);
							coords.push_back(secondhorizontal);
							coords.push_back(secondvertical);
							if (markings.find(selectedkf) == markings.end())
							{
								markings.insert(std::make_pair(selectedkf, framedata()));
							}
							printf("a\n");
							markings[selectedkf].markings.push_back(coords);
							for (auto *fh : keyframes)
							{
								if (fh->id == selectedkf)
								{
									markings[selectedkf].timestamp = fh->timestamp;
									break;
								}
							}
						}
					}
					glPushMatrix();
					glColor3ub(205, 0, 0);

					glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

					glBegin(GL_QUADS);
					glVertex2f(secondhorizontal, firstvertical);
					glVertex2f(firsthorizontal, firstvertical);
					glVertex2f(firsthorizontal, secondvertical);
					glVertex2f(secondhorizontal, secondvertical);
					glEnd();

					//draw previous boxes

					if (markings.find(selectedkf) != markings.end())
					{ // there are boxes to draw
						for (auto coords : markings[selectedkf].markings)
						{
							glBegin(GL_QUADS);
							glVertex2f(coords[2], coords[1]);
							glVertex2f(coords[0], coords[1]);
							glVertex2f(coords[0], coords[3]);
							glVertex2f(coords[2], coords[3]);
							glEnd();
						}
					}

					glPopMatrix();

					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

					if (selectedkf != -1)
					{
						glColor3ub(255, 255, 0);
						mybigfont.Text("Video at selected position").DrawWindow(d_video_player.GetBounds().l, d_video_player.GetBounds().t());
						glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
						texVideoPlayer.RenderToViewportFlipY();
					}
					else
					{
						glColor3ub(255, 255, 0);
						mybigfont.Text("Original Video").DrawWindow(d_video_player.GetBounds().l, d_video_player.GetBounds().t());
						glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
						texVideo.RenderToViewportFlipY();
					}
				}

				if (setting_render_displayDepth)
				{
					d_kfDepth.Activate();
					glColor3ub(255, 0, 0);
					myfont.Text("features").DrawWindow(d_kfDepth.GetBounds().l, d_kfDepth.GetBounds().b - 1.0f * myfont.Height());
					glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
					texKFDepth.RenderToViewportFlipY();
				}

				// if (setting_render_displayResidual )
				// {
				// 	d_residual.Activate();
				// 	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
				// 	texResidual.RenderToViewportFlipY();
				// }


				// update parameters
				this->settings_pointCloudMode = settings_pointCloudMode.Get();

				this->settings_showActiveConstraints = settings_showActiveConstraints.Get();
				this->settings_showAllConstraints = settings_showAllConstraints.Get();
				this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
				this->settings_showKFCameras = settings_showKFCameras.Get();
				this->settings_showTrajectory = settings_showTrajectory.Get();
				this->settings_showFullTrajectory = settings_showFullTrajectory.Get();

				setting_render_display3D = settings_show3D.Get();
				setting_render_displayDepth = settings_showLiveDepth.Get();
				setting_render_displayVideo = settings_showLiveVideo.Get();
				setting_render_displayResidual = settings_showLiveResidual.Get();

				setting_render_renderWindowFrames = settings_showFramesWindow.Get();
				setting_render_plotTrackingFull = settings_showFullTracking.Get();
				setting_render_displayCoarseTrackingFull = settings_showCoarseTracking.Get();

				this->settings_absVarTH = settings_absVarTH.Get();
				this->settings_scaledVarTH = settings_scaledVarTH.Get();
				this->settings_minRelBS = settings_minRelBS.Get();
				this->settings_sparsity = settings_sparsity.Get();

				setting_desiredPointDensity = settings_nPts.Get();
				setting_desiredImmatureDensity = settings_nCandidates.Get();
				setting_maxFrames = settings_nMaxFrames.Get();
				setting_kfGlobalWeight = settings_kfFrequency.Get();
				setting_minGradHistAdd = settings_gradHistAdd.Get();

				if (settings_resetButton.Get())
				{
					printf("RESET!\n");
					settings_resetButton.Reset();
					setting_fullResetRequested = true;
				}

				if (settings_playbackForwardButton.Get())
				{
					printf("FORWARD!\n");
					settings_playbackForwardButton.Reset();
					playback_mode = FORWARD;
				}
				if (settings_playbackReverseButton.Get())
				{
					printf("REVERSE!\n");
					settings_playbackReverseButton.Reset();
					playback_mode = REVERSE;
				}
				if (settings_playbackPauseButton.Get())
				{
					printf("PAUSE!\n");
					settings_playbackPauseButton.Reset();
					playback_mode = PAUSE;
				}
				if (settings_deleteAllMarkings.Get())
				{
					printf("deleting all markings!\n");
					settings_deleteAllMarkings.Reset();
					markings.clear();
					firsthorizontal = 0;
					secondhorizontal = 0;
					firstvertical = 0;
					secondvertical = 0;
				}
				if (settings_deleteMarkings.Get())
				{
					printf("deleting markings from the frame!\n");
					settings_deleteMarkings.Reset();
					markings[selectedkf].markings.clear();
					markings.erase(selectedkf);
					firsthorizontal = 0;
					secondhorizontal = 0;
					firstvertical = 0;
					secondvertical = 0;
				}
				// Swap frames and Process Events
				pangolin::FinishFrame();
				if (selectedkf_index != -1 && updated_frame)
				{
					switch (playback_mode)
					{
					case FORWARD:
						if (selectedkf_index < keyframes.size() - 1)
						{
							if (frameseconds + 1 / settings_playbackFps.Get() < clock() / (float)(CLOCKS_PER_SEC))
							{
								frameseconds = clock() / (float)(CLOCKS_PER_SEC);
								updated_frame = false;
								selectedkf_index++;
								selectedkf = keyframes[selectedkf_index]->id;
								selectedkfchange = true;
								firsthorizontal = 0;
								secondhorizontal = 0;
								firstvertical = 0;
								secondvertical = 0;
							}
						}
						else
							playback_mode = PAUSE;
						// code block
						break;
					case REVERSE:
						if (selectedkf_index > 0)
						{
							if (frameseconds + 1 / settings_playbackFps.Get() < clock() / (float)(CLOCKS_PER_SEC))
							{
								frameseconds = clock() / (float)(CLOCKS_PER_SEC);
								updated_frame = false;
								selectedkf_index--;
								selectedkf = keyframes[selectedkf_index]->id;
								selectedkfchange = true;
								firsthorizontal = 0;
								secondhorizontal = 0;
								firstvertical = 0;
								secondvertical = 0;
							}
						}
						else
							playback_mode = PAUSE;
						// code block
						break;
					default:
						break;
						// code block
					}
				}
				if (needReset)
					reset_internal();
				if (saveimage)
				{
					saveimage = false;
					markings[selectedkf].image = HoldFramebuffer(d_video_player.GetBounds());
					markings[selectedkf].pointcloud = HoldFramebuffer(Visualization3D_display.GetBounds());
				}
				if (settings_saveMarkings.Get())
				{
					printf("saving markings!\n");
					settings_saveMarkings.Reset();
					std::map<int, framedata>::iterator it;
					pangolin::PixelFormat fmt = pangolin::PixelFormatFromString("RGBA32");

					for (it = markings.begin(); it != markings.end(); it++)
					{
						int mins= (int)((it->second.timestamp / 1000000000)/60);
						int secs = (it->second.timestamp / 1000000000)-mins*60;
						int msecs = (it->second.timestamp / 1000000) - secs * 1000;

						pangolin::SaveImage(it->second.pointcloud, fmt, str(boost::format("save/pointcloud_%dmin_%ds_%dms") % mins % secs % msecs) + ".png", false);
						pangolin::SaveImage(it->second.image, fmt, str(boost::format("save/image_%dmin_%ds_%dms") % mins % secs % msecs) + ".png", false);
					}
				}
			}
			printf("QUIT Pangolin thread!\n");
			// printf("I'll just kill the whole process.\nSo Long, and Thanks for All the Fish!\n");

			exit(1);
		}

		void PangolinDSOViewer::close()
		{
			running = false;
		}

		void PangolinDSOViewer::join()
		{
			runThread.join();
			printf("JOINED Pangolin thread!\n");
		}

		void PangolinDSOViewer::reset()
		{
			needReset = true;
		}

		void PangolinDSOViewer::reset_internal()
		{
			model3DMutex.lock();
			for (size_t i = 0; i < keyframes.size(); i++)
				delete keyframes[i];
			keyframes.clear();
			allFramePoses.clear();
			keyframesByKFID.clear();
			connections.clear();
			model3DMutex.unlock();

			openImagesMutex.lock();
			internalVideoImg->setBlack();
			internalVideoPlayerImg->setBlack();
			internalKFImg->setBlack();
			// internalResImg->setBlack();
			videoImgChanged = kfImgChanged = resImgChanged = videoPlayerImgChanged = true;
			openImagesMutex.unlock();

			needReset = false;
		}

		void PangolinDSOViewer::drawConstraints()
		{
			if (settings_showAllConstraints)
			{
				// draw constraints
				glLineWidth(1);
				glBegin(GL_LINES);

				glColor3f(0, 1, 0);
				glBegin(GL_LINES);
				for (unsigned int i = 0; i < connections.size(); i++)
				{
					if (connections[i].to == 0 || connections[i].from == 0)
						continue;
					int nAct = connections[i].bwdAct + connections[i].fwdAct;
					int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;
					if (nAct == 0 && nMarg > 0)
					{
						Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
						glVertex3f((GLfloat)t[0], (GLfloat)t[1], (GLfloat)t[2]);
						t = connections[i].to->camToWorld.translation().cast<float>();
						glVertex3f((GLfloat)t[0], (GLfloat)t[1], (GLfloat)t[2]);
					}
				}
				glEnd();
			}

			if (settings_showActiveConstraints)
			{
				glLineWidth(3);
				glColor3f(0, 0, 1);
				glBegin(GL_LINES);
				for (unsigned int i = 0; i < connections.size(); i++)
				{
					if (connections[i].to == 0 || connections[i].from == 0)
						continue;
					int nAct = connections[i].bwdAct + connections[i].fwdAct;

					if (nAct > 0)
					{
						Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
						glVertex3f((GLfloat)t[0], (GLfloat)t[1], (GLfloat)t[2]);
						t = connections[i].to->camToWorld.translation().cast<float>();
						glVertex3f((GLfloat)t[0], (GLfloat)t[1], (GLfloat)t[2]);
					}
				}
				glEnd();
			}

			if (settings_showTrajectory)
			{
				float colorRed[3] = {1, 0, 0};
				glColor3f(colorRed[0], colorRed[1], colorRed[2]);
				glLineWidth(3);

				glBegin(GL_LINE_STRIP);
				for (unsigned int i = 0; i < keyframes.size(); i++)
				{
					glVertex3f((float)keyframes[i]->camToWorld.translation()[0],
							   (float)keyframes[i]->camToWorld.translation()[1],
							   (float)keyframes[i]->camToWorld.translation()[2]);
				}
				glEnd();
			}

			if (settings_showFullTrajectory)
			{
				float colorGreen[3] = {0, 1, 0};
				glColor3f(colorGreen[0], colorGreen[1], colorGreen[2]);
				glLineWidth(3);

				glBegin(GL_LINE_STRIP);
				for (unsigned int i = 0; i < allFramePoses.size(); i++)
				{
					glVertex3f((float)allFramePoses[i][0],
							   (float)allFramePoses[i][1],
							   (float)allFramePoses[i][2]);
				}
				glEnd();
			}
		}

		void PangolinDSOViewer::publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity)
		{
			if (!setting_render_display3D)
				return;
			if (disableAllDisplay)
				return;

			model3DMutex.lock();
			connections.resize(connectivity.size());
			int runningID = 0;
			int totalActFwd = 0, totalActBwd = 0, totalMargFwd = 0, totalMargBwd = 0;
			for (std::pair<uint64_t, Eigen::Vector2i> p : connectivity)
			{
				int host = (int)(p.first >> 32);
				int target = (int)(p.first & (uint64_t)0xFFFFFFFF);

				assert(host >= 0 && target >= 0);
				if (host == target)
				{
					assert(p.second[0] == 0 && p.second[1] == 0);
					continue;
				}

				if (host > target)
					continue;

				connections[runningID].from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
				connections[runningID].to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
				connections[runningID].fwdAct = p.second[0];
				connections[runningID].fwdMarg = p.second[1];
				totalActFwd += p.second[0];
				totalMargFwd += p.second[1];

				uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
				Eigen::Vector2i st = connectivity.at(inverseKey);
				connections[runningID].bwdAct = st[0];
				connections[runningID].bwdMarg = st[1];

				totalActBwd += st[0];
				totalMargBwd += st[1];

				runningID++;
			}

			model3DMutex.unlock();
		}
		void PangolinDSOViewer::publishKeyframes(
			std::vector<FrameHessian *> &frames,
			bool final,
			CalibHessian *HCalib)
		{
			if (!setting_render_display3D)
				return;
			if (disableAllDisplay)
				return;

			boost::unique_lock<boost::mutex> lk(model3DMutex);
			for (FrameHessian *fh : frames)
			{
				if (keyframesByKFID.find(fh->frameID) == keyframesByKFID.end())
				{
					KeyFrameDisplay *kfd = new KeyFrameDisplay();
					keyframesByKFID[fh->frameID] = kfd;
					keyframes.push_back(kfd);
				}
				keyframesByKFID[fh->frameID]->setFromKF(fh, HCalib);
				keyframesByKFID[fh->frameID]->timestamp = fh->shell->timestamp;
			}
		}
		void PangolinDSOViewer::publishCamPose(FrameShell *frame,
											   CalibHessian *HCalib)
		{
			if (!setting_render_display3D)
				return;
			if (disableAllDisplay)
				return;

			boost::unique_lock<boost::mutex> lk(model3DMutex);
			struct timeval time_now;
			gettimeofday(&time_now, NULL);
			lastNTrackingMs.push_back(((time_now.tv_sec - last_track.tv_sec) * 1000.0f + (time_now.tv_usec - last_track.tv_usec) / 1000.0f));
			if (lastNTrackingMs.size() > 10)
				lastNTrackingMs.pop_front();
			last_track = time_now;

			if (!setting_render_display3D)
				return;

			currentCam->setFromF(frame, HCalib);
			allFramePoses.push_back(frame->camToWorld.translation().cast<float>());
		}

		void PangolinDSOViewer::pushLiveFrame(FrameHessian *image)
		{
			if (!setting_render_displayVideo)
				return;
			if (disableAllDisplay)
				return;

			boost::unique_lock<boost::mutex> lk(openImagesMutex);

			for (int i = 0; i < w * h; i++)
				internalVideoImg->data[i][0] =
					internalVideoImg->data[i][1] =
						internalVideoImg->data[i][2] =
							image->dI[i][0] * 0.8 > 255.0f ? 255.0 : image->dI[i][0] * 0.8;

			videoImgChanged = true;
		}
		void PangolinDSOViewer::pushLiveFrame(ColorImageAndExposure *image)
		{
			if (!setting_render_displayVideo)
				return;
			if (disableAllDisplay)
				return;

			boost::unique_lock<boost::mutex> lk(openImagesMutex);

			for (int i = 0; i < w * h; i++)
			{
				internalVideoImg->data[i][0] =
					image->image[i][0] > 255.0f ? 255.0 : image->image[i][0];
				internalVideoImg->data[i][1] =
					image->image[i][1] > 255.0f ? 255.0 : image->image[i][1];
				internalVideoImg->data[i][2] =
					image->image[i][2] > 255.0f ? 255.0 : image->image[i][2];
			}
			videoImgChanged = true;
		}

		void PangolinDSOViewer::pushRequestedFrame(ColorImageAndExposure *image)
		{
			if (!setting_render_displayVideo)
				return;
			if (disableAllDisplay)
				return;
			boost::unique_lock<boost::mutex> lk(openImagesMutex);

			for (int i = 0; i < w * h; i++)
			{
				internalVideoPlayerImg->data[i][0] =
					image->image[i][0] > 255.0f ? 255.0 : image->image[i][0];
				internalVideoPlayerImg->data[i][1] =
					image->image[i][1] > 255.0f ? 255.0 : image->image[i][1];
				internalVideoPlayerImg->data[i][2] =
					image->image[i][2] > 255.0f ? 255.0 : image->image[i][2];
			}
			videoPlayerImgChanged = true;
		}

		bool PangolinDSOViewer::needPushDepthImage()
		{
			return setting_render_displayDepth;
		}
		void PangolinDSOViewer::pushDepthImage(MinimalImageB3 *image)
		{

			if (!setting_render_displayDepth)
				return;
			if (disableAllDisplay)
				return;

			boost::unique_lock<boost::mutex> lk(openImagesMutex);

			struct timeval time_now;
			gettimeofday(&time_now, NULL);
			lastNMappingMs.push_back(((time_now.tv_sec - last_map.tv_sec) * 1000.0f + (time_now.tv_usec - last_map.tv_usec) / 1000.0f));
			if (lastNMappingMs.size() > 10)
				lastNMappingMs.pop_front();
			last_map = time_now;

			memcpy(internalKFImg->data, image->data, w * h * 3);
			kfImgChanged = true;
		}

	}
}
