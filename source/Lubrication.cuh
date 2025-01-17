// Maintainer: Andrew Fiore

/*! \file Lubrication.cuh
    \brief Define the GPU kernels and driving functions to compute the Lubrication
	interactions. 
*/

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

// Edmond 03/31/2023: try to mimic PPPMForceComputeGPU & CommunicatorGridGPU for HIP
#include "hip/hip_runtime.h"

#if defined(ENABLE_HIP)
#ifdef __HIP_PLATFORM_HCC__
#include <hipfft.h>
#else
#include <cufft.h>
typedef cufftComplex hipfftComplex;
#endif
#endif
/*********************/
// #include <cufft.h>

#include <thrust/version.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cusparse.h>
#include <cusolverSp.h>

//! Define the step_one kernel
#ifndef __LUBRICATION_CUH__
#define __LUBRICATION_CUH__

// Edmond 03/31/2023: try to mimic PPPMForceComputeGPU & CommunicatorGridGPU for HIP
//! Definition for comxplex variable storage
#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX hipfftComplex
#else
#define CUFFTCOMPLEX hipfftComplex
#endif

// //! Definition for comxplex variable storage
// #ifdef SINGLE_PRECISION
// #define CUFFTCOMPLEX cufftComplex
// #else
// #define CUFFTCOMPLEX cufftComplex
// #endif

namespace hoomd
{

__global__ void Lubrication_RFU_kernel(
					Scalar *d_AppliedForce,   // output
					const Scalar *d_Velocity, // input
					const Scalar4 *d_pos,
					unsigned int *d_group_members,
					const int group_size, 
			      		const BoxDim box,
					const unsigned int *d_n_neigh, 
					unsigned int *d_nlist, 
					const long unsigned int *d_headlist, 
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const Scalar ResTable_min,
					const Scalar ResTable_dr,
					const Scalar rlub
					);

__global__ void Lubrication_RFE_kernel(
					Scalar *d_Force,
					Scalar shear_rate,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size, 
			      		BoxDim box,
					const unsigned int *d_n_neigh, 
					unsigned int *d_nlist, 
					const long unsigned int *d_headlist, 
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const Scalar ResTable_min,
					const Scalar ResTable_dr,
					const Scalar rlub
					);

__global__ void Lubrication_RSU_kernel(
					Scalar *d_Stresslet,
					Scalar *d_Velocity,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					int group_size, 
			      		BoxDim box,
					const unsigned int *d_n_neigh, 
					unsigned int *d_nlist, 
					const long unsigned int *d_headlist, 
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const Scalar ResTable_min,
					const Scalar ResTable_dr,
					const Scalar rlub
					);

__global__ void Lubrication_RSE_kernel(
					Scalar *d_Stresslet,
					Scalar strain_rate,
					int group_size, 
					unsigned int *d_group_members,
					const unsigned int *d_n_neigh, 
					unsigned int *d_nlist, 
					const long unsigned int *d_headlist, 
					Scalar4 *d_pos,
			      		BoxDim box,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const Scalar ResTable_min,
					const Scalar ResTable_dr
					);

__global__ void Lubrication_RSEgeneral_kernel(
					Scalar *d_Stresslet,
					Scalar *d_Strain,
					int group_size, 
					unsigned int *d_group_members,
					const unsigned int *d_n_neigh, 
					unsigned int *d_nlist, 
					const long unsigned int *d_headlist, 
					Scalar4 *d_pos,
			      		BoxDim box,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const Scalar ResTable_min,
					const Scalar ResTable_dr
					);

}
#endif
