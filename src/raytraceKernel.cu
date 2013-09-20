// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define M_PI 3.14159265359.0f

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  	// Create ray using pinhole camera projection
  float px_size_x = tan( fov.x * (PI/180.0) );
  float px_size_y = tan( fov.y * (PI/180.0) );
	
  ray r;
  r.origin = eye;
  r.direction = view + (-2*px_size_x*x/resolution.x + px_size_x)*glm::cross( view, up ) \
				     + (-2*px_size_y*y/resolution.y + px_size_y)*up;

  return r;
}

__host__ __device__ glm::vec3 computeGeomColor( int min_intersection_ind, staticGeom* geoms, material* materials, ray r, float intersection_dist, glm::vec3 intersection_normal, glm::vec3 intersection_point ) {
	// Set color equal to material color
	int mat_id = geoms[min_intersection_ind].materialid;
	//colors[index] = materials[mat_id].color;

	// Calculate Phong Lighting
	// Start with arbritrarily chosen light source, lets say from the camera
		
	glm::vec3 debug_light_source(0.0, 0.0, 0.0);
	glm::vec3 light_vector = glm::normalize(debug_light_source - intersection_point);

	glm::vec3 viewing_vector = glm::normalize( r.origin - intersection_point ); 
	glm::vec3 reflection_vector = 2*glm::dot( light_vector, intersection_normal )*intersection_normal - light_vector;
		
	// Calculate Phong Reflection Model ... this is mad inefficient at the moment
	//float m_d = 1.0; //?
	//float s_d = 1.0; //?
	//float c_d = s_d*m_d*max( glm::dot( intersection_normal, light_vector ), 0.0 );
	//float c_d = powf( max( glm::dot( light_vector, viewing_vector ), 0.0 ), materials[mat_id].specularExponent );
	float ks = 1.0; // specular reflection constant
	float kd = 0.5; // diffuse reflection constant
	float ka = 0.5; // ambient reflection constant

	// Ambient Component
	glm::vec3 ambient(1.0, 1.0, 1.0);

	// Diffuse Component
	//glm::vec3 diffuseIntensity( 1.0, 1.0, 1.0 ); Not needed at the moment
	float diffuse = max(glm::dot( light_vector, intersection_normal ), 0.0);

	// Specular Component  
	float specularExponent = materials[mat_id].specularExponent; // alpha, shinyiness
	glm::vec3 specColor = materials[mat_id].specularColor;
	glm::vec3 specular( 0.0, 0.0, 0.0 );
		
	if ( specularExponent > 0.0 ) {
		specular = specColor*powf( max( glm::dot( reflection_vector, viewing_vector ), 0.0 ), specularExponent );
	} 
		
	// Full illumination
	glm::vec3 Illumination = ka*ambient + kd*diffuse + ks*specular;
	return Illumination*materials[mat_id].color;
}

// Compute Light Contribution to object
__host__ __device__ glm::vec3 computeLightContribution(  material mat, ray current_ray, ray light_ray, glm::vec3 intersection_normal, glm::vec3 intersection_point ) {
	glm::vec3 light_vector = light_ray.direction;
	glm::vec3 viewing_vector = glm::normalize( current_ray.origin - intersection_point ); 
	glm::vec3 reflection_vector = 2*glm::dot( light_vector, intersection_normal )*intersection_normal - light_vector;
	
	// Temporarily
	float ka = 0.5; // ambient 
	float ks = 1.0; // specular reflection constant
	float kd = 0.5; // diffuse reflection constant

	glm::vec3 ambient( 1.0, 1.0, 1.0 );

	float diffuse = max(glm::dot( light_vector, intersection_normal ), 0.0);

	// Specular Component  
	float specularExponent = mat.specularExponent; // alpha, shinyiness
	glm::vec3 specColor = mat.specularColor;
	glm::vec3 specular( 0.0, 0.0, 0.0 );
		
	if ( specularExponent > 0.0 ) {
		specular = specColor*powf( max( glm::dot( reflection_vector, viewing_vector ), 0.0 ), specularExponent );
	} 
		
	// Full illumination
	glm::vec3 illumination = ka*ambient + kd*diffuse + ks*specular;
	return illumination*mat.color;
}

// Find closest intersection
__host__ __device__ int closestIntersection( ray r, staticGeom* geoms, int numberOfGeoms, float& intersection_dist, glm::vec3& intersection_normal, glm::vec3& intersection_point ) {
	// Check for intersections. This has way too many branches :/
	int min_intersection_ind = -1;
	float intersection_dist_new;
	glm::vec3 intersection_point_new;
	glm::vec3 intersection_normal_new;

	for (int i=0; i < numberOfGeoms; ++i ) {
	    // Check for intersection with Sphere
		if ( geoms[i].type == SPHERE ) {
		    intersection_dist_new = sphereIntersectionTest(geoms[i], r, intersection_point_new, intersection_normal_new);		
						
		} else if ( geoms[i].type == CUBE ) {
			intersection_dist_new = boxIntersectionTest(geoms[i], r, intersection_point_new, intersection_normal_new);		
		} else if ( geoms[i].type == MESH ) {
			// TODO
		}
		if (intersection_dist_new != -1 ) {
			
			// If new distance is closer than previously seen one then use the new one
			if ( intersection_dist_new < intersection_dist || intersection_dist == -1 ) {
				intersection_dist = intersection_dist_new;
				intersection_point = intersection_point_new;
				intersection_normal = intersection_normal_new;
				min_intersection_ind = i;
			}
		}	
	}
	return min_intersection_ind;
}

// Check if ray to light is occluded by an object
// This is going to be super inefficient, for each geom check if its a light if so trace a ray to 
// it and see if that intersects with any other geoms. 
__host__ __device__ int isShadowRay( glm::vec3 light, ray &light_ray, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, glm::vec3 intersection_point ) {
	// DOESN'T WORK YET!?!?
	
	// Ray to light
    light_ray.origin = intersection_point;
	
	int obj_ind = -1;
	
	glm::vec3 light_vector;
	
	// Unfortunately I don't really care about these, I should probably do a refactor
	glm::vec3 obstacle_intersection_normal;
	glm::vec3 obstacle_intersection_point;
	float obstacle_intersection_dist;
	
	// Closest light index
	int light_index = -1;

	light_ray.direction = glm::normalize(light - intersection_point);

	ray intersection_ray;
	intersection_ray.origin = intersection_point;
	intersection_ray.direction = light_ray.direction;

	obj_ind = closestIntersection( intersection_ray, geoms, numberOfGeoms, obstacle_intersection_dist, obstacle_intersection_normal, obstacle_intersection_point );
	
	return obj_ind;
}	

// Calculate reflected ray
__host__ __device__ ray computeReflectedRay( ray currentRay, glm::vec3 intersection_normal, glm::vec3 intersection_point ) {
	ray reflected_ray;
	reflected_ray.origin = intersection_point;
	reflected_ray.direction = -2*glm::dot(currentRay.direction, intersection_normal)*intersection_normal + currentRay.direction;
	return reflected_ray;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  int obj_index = -1;
  float intersection_dist = -1; // where is NaN / Infinity? dumb shit
  glm::vec3 intersection_point;
  glm::vec3 intersection_normal;
  glm::vec3 intersection_point_new;
  glm::vec3 intersection_normal_new;

  // Ambient Component
  //glm::vec3 ambient(0.2, 0.2, 0.5);
  glm::vec3 ambient(0.0, 0.0, 0.0);

  //glm::vec3 light( 10.0, 0.0, 0.0 );
  glm::vec3 light(-5.0, 0.0, -2.0);
  //glm::vec3 color(0.0, 0.0, 0.0);
  glm::vec3 color = ambient;

  glm::vec3 colorContribution(1.0,1.0,1.0);

  if((x<=resolution.x && y<=resolution.y)){
	  
	// Calculate initial ray as projected from camera
	ray currentRay = raycastFromCameraKernel( cam.resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov );
	ray lightRay;

	// Iteratively trace rays until depth is reached
	int depth = 4;
	for (int i=0; i<depth; ++i) {
		obj_index = closestIntersection( currentRay, geoms, numberOfGeoms, intersection_dist, intersection_normal, intersection_point );
		if (obj_index == -1) {
			break;
		}

		//int mat_id = isShadowRay( lightRay,  geoms, numberOfGeoms, materials, numberOfMaterials ); 
		int mat_id = isShadowRay( light, lightRay, geoms, numberOfGeoms, materials, numberOfMaterials, intersection_point );
		if ( mat_id == -1 ) { 
			color += colorContribution*computeLightContribution( materials[geoms[obj_index].materialid], currentRay, lightRay, intersection_normal, intersection_point );
			colorContribution *= materials[geoms[obj_index].materialid].absorptionCoefficient;
		}
		// Calculate reflected rays
		if ( materials[geoms[obj_index].materialid].hasReflective ) {
			currentRay = computeReflectedRay( currentRay, intersection_normal, intersection_point );
		}
			
	}
    //colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
	colors[index] = color;
  }

}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  //package geometry and materials and sent to GPU
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
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();


  cudaError_t errorNum = cudaPeekAtLastError();
  if ( errorNum != cudaSuccess ) { 
      printf ("Cuda error -- %s\n", cudaGetErrorString(errorNum));
  }
  checkCUDAError("Kernel failed!");
}
