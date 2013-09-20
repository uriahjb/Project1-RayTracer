// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ cudaMat4 removeScale( cudaMat4 m, glm::vec3 scale );
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
    if(fabs(fabs(a)-fabs(b))<EPSILON){
        return true;
    }else{
        return false;
    }
}

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t-.0001)*glm::normalize(r.direction);
}

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

// Remove the scale from the transformation matrix so that we can perform 
// matrix operations on vectors etc without having the scale of these 
// vectors messed with 
__host__ __device__ cudaMat4 removeScale( cudaMat4 m, glm::vec3 scale ) {
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

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	/*
		Compute cube intersection using line-plane intersection as per wikipedia
		en.wikipedia/wiki/Line-Plane_Intersection

		For each normal of the plane compute:
			d = dot((p0 - l0), n) / dot(l, n) checking for parallel and in-plane conditions
		
		Find the closest intersection point and check if it is within the x-y-z bounds of the cube
	*/

	// Convert global ray coordinates to local box coordinates   
	cudaMat4 inv_tf_no_scale = removeScale( box.inverseTransform, glm::vec3( 1/box.scale.x, 1/box.scale.y, 1/box.scale.z));

	glm::vec3 ro = multiplyMV(inv_tf_no_scale, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(inv_tf_no_scale, glm::vec4(r.direction,0.0f)));
	ray rt; rt.origin = ro; rt.direction = rd;
	
	bool is_valid = false;
	float t;
	float min_t = 0;
	glm::vec3 face_pos;
	glm::vec3 face_pos_norm;

	for (int i=0; i < 6; ++i ) {
		//glm::vec3 face_pos(  0.0,  0.0,  0.0 );
		//glm::vec3 face_pos_norm( 0.0,  0.0,  1.0);
		int ind = i % 3;
		face_pos = glm::vec3( 0.0, 0.0, 0.0 );
		face_pos_norm = glm::vec3( 0.0, 0.0, 0.0 );
		if ( i < 3 ) {
			face_pos[ind] = box.scale[ind]/2;
			face_pos_norm[ind] = 1.0;
		} else {
			face_pos[ind] = -box.scale[ind]/2;
			face_pos_norm[ind] = -1.0;
		}
		float den = glm::dot( rt.direction, face_pos_norm );
		float num = glm::dot( face_pos - rt.origin, face_pos_norm );

		float tol = 1e-6;
		if ( abs(num) < tol ) {
			continue;
		}
		if ( abs(den) < tol ) {
			continue;
		}
		t = num/den;
		// add 0.001 to account for the 0.001 subtracted in getPointOnRay
		glm::vec3 localIntersectionPoint = getPointOnRay( rt, t + 0.001 );
		if ( localIntersectionPoint.x >= box.scale.x/2 || localIntersectionPoint.x <= -box.scale.x/2 
		  || localIntersectionPoint.y >= box.scale.y/2 || localIntersectionPoint.y <= -box.scale.y/2
		  || localIntersectionPoint.z >= box.scale.z/2 || localIntersectionPoint.z <= -box.scale.z/2) {
			continue;
		}
		min_t = min(t, min_t);
		is_valid = true;
	}
	if (!is_valid) {
		return -1;
	}

	//float t = 0.0;
	cudaMat4 tf_no_scale = removeScale( box.transform, box.scale );
	glm::vec3 realIntersectionPoint = multiplyMV(tf_no_scale, glm::vec4(getPointOnRay(rt, t), 1.0));
    glm::vec3 realOrigin = multiplyMV(tf_no_scale, glm::vec4(0,0,0,1));
	
    intersectionPoint = realIntersectionPoint;
    normal = glm::normalize(realIntersectionPoint - realOrigin);

	return t;
    //return glm::length(r.origin - realIntersectionPoint);
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1;
  } else if (t1 > 0 && t2 > 0) {
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
  }

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);
        
  return glm::length(r.origin - realIntersectionPoint);
}

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

//LOOK: Example for generating a random point on an object using thrust.
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    //get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f; //x-y face
    float side2 = radii.z * radii.y * 4.0f; //y-z face
    float side3 = radii.x * radii.z* 4.0f; //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    //pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        //x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        //x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        //y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        //y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        //x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        //x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

  return glm::vec3(0,0,0);
}

#endif


