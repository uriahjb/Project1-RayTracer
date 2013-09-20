// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  #ifdef __APPLE__
	  // Needed in OSX to force use of OpenGL3.2 
	  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
	  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
	  glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	  glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #endif

  // Set up pathtracer stuff
  bool loadedScene = false;
  finishedRender = false;

  targetFrame = 0;
  singleFrameMode = false;

  // Load scene file
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "scene")==0){
      renderScene = new scene(data);
      loadedScene = true;
    }else if(strcmp(header.c_str(), "frame")==0){
      targetFrame = atoi(data.c_str());
      singleFrameMode = true;
    }
  }

  if(!loadedScene){
    cout << "Error: scene file needed!" << endl;
    return 0;
  }

  // Set up camera stuff from loaded pathtracer settings
  iterations = 0;
  renderCam = &renderScene->renderCam;
  width = renderCam->resolution[0];
  height = renderCam->resolution[1];

  if(targetFrame>=renderCam->frames){
    cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
    targetFrame = 0;
  }

  // Launch CUDA/GL

  #ifdef __APPLE__
	init();
  #else
	init(argc, argv);
  #endif

  initCuda();

  initVAO();
  initTextures();

  GLuint passthroughProgram;
  passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  #ifdef __APPLE__
	  // send into GLFW main loop
	  while(1){
		display();
		if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS || !glfwGetWindowParam( GLFW_OPENED )){
				exit(0);
		}
	  }

	  glfwTerminate();
  #else
	  glutDisplayFunc(display);
	  glutKeyboardFunc(keyboard);

	  glutMainLoop();
  #endif
  return 0;
}

//Some forward declarations
__host__ __device__ glm::vec3 multiplyMVDbg(cudaMat4 m, glm::vec4 v);

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMVDbg(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

__host__ __device__ cudaMat4 removeScaleDbg( cudaMat4 m, glm::vec3 scale ) {
	// The scale is either distributed along each row of the rotation matrix
	// or along each column. I'm not entirely sure which yet. 
	cudaMat4 n = m;
	n.x.x = m.x.x/scale[0];
	n.x.y = m.x.y/scale[0];
	n.x.z = m.x.z/scale[0];
	n.y.x = m.y.x/scale[1];
	n.y.y = m.y.y/scale[1];
	n.y.z = m.y.z/scale[1];
	n.z.x = m.z.x/scale[2];
	n.z.y = m.z.y/scale[2];
	n.z.z = m.z.z/scale[2];
	return n;
}

void printMat4( cudaMat4 mat ) {
	int i,j;
	printf( "[[%f, %f, %f, %f]\n,[%f,%f,%f, %f]\n,[%f,%f,%f, %f],\n[%f, %f, %f, %f]]\n", \
			 mat.x.x, mat.x.y, mat.x.z, mat.x.w, \
			 mat.y.x, mat.y.y, mat.y.z, mat.y.w, \
			 mat.z.x, mat.z.y, mat.z.z, mat.z.w, \
			 mat.w.x, mat.w.y, mat.w.z, mat.w.w );		
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){

  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  
  if(iterations<renderCam->iterations){
    uchar4 *dptr=NULL;
    iterations++;
    cudaGLMapBufferObject((void**)&dptr, pbo);
  
    //pack geom and material arrays
    geom* geoms = new geom[renderScene->objects.size()];
    material* materials = new material[renderScene->materials.size()];
    
    for(int i=0; i<renderScene->objects.size(); i++){
      geoms[i] = renderScene->objects[i];
    }
    for(int i=0; i<renderScene->materials.size(); i++){
      materials[i] = renderScene->materials[i];
    }
    
	// DEBUGGING
	/*
	ray r;
	camera cam = *renderCam;
	int x = 400;
	int y = 400;
	float px_size_x = tan( cam.fov.x * (PI/180.0) );
	float px_size_y = tan( cam.fov.y * (PI/180.0) );

	r.direction = cam.views[0] + px_size_x*x/cam.resolution.x*glm::cross( cam.views[0], cam.ups[0] ) \
							   + px_size_y*y/cam.resolution.y*cam.ups[0];
	printf( "Camera view: [%f, %f, %f] \n\r", cam.views[0].x, cam.views[0].y, cam.views[0].z );
	printf( "Camera fov: [%f, %f] \n\r", cam.fov.x, cam.fov.y ); 
	printf( "ray direction: [%f, %f, %f] \n\r", r.direction.x, r.direction.y, r.direction.z);
    */
    // DEBUGGING LINE-PLANE INTERSECTION
	// Pull box outta geoms
	/*
	  //package geometry and materials and sent to GPU
	  int numberOfGeoms = renderScene->objects.size();
	  int frame = 0;
	  staticGeom* geomList = new staticGeom[numberOfGeoms];
	  for(int i=0; i<numberOfGeoms; i++){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[i] = newStaticGeom;
	  }

	
    camera cam = *renderCam;
	
	int x = 200;
	int y = 200;

	staticGeom box = geomList[3];
	float px_size_x = tan( cam.fov.x * (PI/180.0) );
	float px_size_y = tan( cam.fov.y * (PI/180.0) );
	
	ray r;
	r.origin = cam.positions[0];
	r.direction = cam.views[0] + (2*px_size_x*x/cam.resolution.x - px_size_x)*glm::cross( cam.positions[0], cam.ups[0] ) \
       				           + (2*px_size_y*y/cam.resolution.y - px_size_y)*cam.ups[0];
	// There are few things more frustrating than a transformation matrix that has the 
	// scale rolled into it. 
	printf( "box tf: \n" );
	printMat4( box.transform );
	printf( "box tf no scale: \n" );
	printMat4( removeScaleDbg(box.transform, box.scale) );
	printf( "box inv tf: \n");
	printMat4( box.inverseTransform );
	printf( "box inv tf no scale: \n");
	printMat4( removeScaleDbg(box.inverseTransform, glm::vec3( 1/box.scale.x, 1/box.scale.y, 1/box.scale.z)) );

	
	//glm::vec3 ro = multiplyMVDbg(box.inverseTransform, glm::vec4(r.origin,1.0f));
	//glm::vec3 rd = glm::normalize(multiplyMVDbg(box.inverseTransform, glm::vec4(r.direction,0.0f)));

	cudaMat4 inv_tf_no_scale = removeScaleDbg( box.inverseTransform,  glm::vec3( 1/box.scale.x, 1/box.scale.y, 1/box.scale.z));

	glm::vec3 ro = multiplyMVDbg(inv_tf_no_scale, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMVDbg(inv_tf_no_scale, glm::vec4(r.direction,0.0f)));

	ray rt; rt.origin = ro; rt.direction = rd;

	glm::vec3 zf_pos(  0.0,  0.0,  0.0 );
	glm::vec3 zf_pos_norm( 0.0,  0.0,  1.0);
	float den = glm::dot( rt.direction, zf_pos_norm );
	float num = glm::dot( zf_pos - rt.origin, zf_pos_norm );

	printf( "box position: [%f, %f, %f] \n", box.translation.x, box.translation.y, box.translation.z ); 
	printf( "ray origin: [%f, %f, %f] \n", r.origin.x, r.origin.y, r.origin.z );
    printf( "ray direction: [%f, %f, %f] \n", r.direction.x, r.direction.y, r.direction.z );
	printf( "ray tf origin: [%f, %f, %f] \n", rt.origin.x, rt.origin.y, rt.origin.z );
	printf( "ray tf direction: [%f, %f, %f] \n", rt.direction.x, rt.direction.y, rt.direction.z );
	printf( "den: %f \n", den );
	printf( "num: %f \n", num );
	float tol = 1e-6;
	if (abs(num) > tol && abs(den) > tol) {
		float d = num/den;
		glm::vec3 intersection_point = rt.origin + d*glm::normalize( rt.direction );
		printf( "intersection_point:  [%f, %f, %f] \n", intersection_point.x, intersection_point.y, intersection_point.z ); 
	}
	*/


    // execute the kernel
    cudaRaytraceCore(dptr, renderCam, targetFrame, iterations, materials, renderScene->materials.size(), geoms, renderScene->objects.size() );
    
    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
  }else{

    if(!finishedRender){
      //output image file
      image outputImage(renderCam->resolution.x, renderCam->resolution.y);

      for(int x=0; x<renderCam->resolution.x; x++){
        for(int y=0; y<renderCam->resolution.y; y++){
          int index = x + (y * renderCam->resolution.x);
          outputImage.writePixelRGB(renderCam->resolution.x-1-x,y,renderCam->image[index]);
        }
      }
      
      gammaSettings gamma;
      gamma.applyGamma = true;
      gamma.gamma = 1.0/2.2;
      gamma.divisor = renderCam->iterations;
      outputImage.setGammaSettings(gamma);
      string filename = renderCam->imageName;
      string s;
      stringstream out;
      out << targetFrame;
      s = out.str();
      utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
      utilityCore::replaceString(filename, ".png", "."+s+".png");
      outputImage.saveImageRGB(filename);
      cout << "Saved frame " << s << " to " << filename << endl;
      finishedRender = true;
      if(singleFrameMode==true){
        cudaDeviceReset(); 
        exit(0);
      }
    }
    if(targetFrame<renderCam->frames-1){

      //clear image buffer and move onto next frame
      targetFrame++;
      iterations = 0;
      for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
        renderCam->image[i] = glm::vec3(0,0,0);
      }
      cudaDeviceReset(); 
      finishedRender = false;
    }
  }
  
}

#ifdef __APPLE__

	void display(){
		runCuda();

		string title = "CIS565 Render | " + utilityCore::convertIntToString(iterations) + " Iterations";
		glfwSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glfwSwapBuffers();
	}

#else

	void display(){
		runCuda();

		string title = "565Raytracer | " + utilityCore::convertIntToString(iterations) + " Iterations";
		glutSetWindowTitle(title.c_str());

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			  GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glClear(GL_COLOR_BUFFER_BIT);   

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		glutPostRedisplay();
		glutSwapBuffers();
	}

	void keyboard(unsigned char key, int x, int y)
	{
		std::cout << key << std::endl;
		switch (key) 
		{
		   case(27):
			   exit(1);
			   break;
		}
	}

#endif




//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init(){

		if (glfwInit() != GL_TRUE){
			shut_down(1);      
		}

		// 16 bit color, no depth, alpha or stencil buffers, windowed
		if (glfwOpenWindow(width, height, 5, 6, 5, 0, 0, 0, GLFW_WINDOW) != GL_TRUE){
			shut_down(1);
		}

		// Set up vertex array object, texture stuff
		initVAO();
		initTextures();
	}
#else
	void init(int argc, char* argv[]){
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(width, height);
		glutCreateWindow("565Raytracer");

		// Init GLEW
		glewInit();
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			/* Problem: glewInit failed, something is seriously wrong. */
			std::cout << "glewInit failed, aborting." << std::endl;
			exit (1);
		}

		initVAO();
		initTextures();
	}
#endif

void initPBO(GLuint* pbo){
  if (pbo) {
    // set up vertex data parameter
    int num_texels = width*height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);

  runCuda();
}

void initTextures(){
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
    GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  #ifdef __APPLE__
	glfwTerminate();
  #endif
  exit(return_code);
}
