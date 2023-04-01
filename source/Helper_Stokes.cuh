// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Stokes.cuh
    \brief Declares GPU kernel code for helper functions integration considering hydrodynamic interactions on the GPU. Used by Stokes.
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

//! Define the step_one kernel
#ifndef __HELPER_STOKES_CUH__
#define __HELPER_STOKES_CUH__

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

using namespace hoomd;

__global__ void Stokes_SetForce_kernel(
					Scalar4 *d_net_force,
					float   *d_AppliedForce,
					unsigned int group_size,
					unsigned int *d_group_members
					);

__global__ void Stokes_SetForce_manually_kernel(
						const Scalar4 *d_pos,
						float   *d_AppliedForce,
						unsigned int group_size,
						unsigned int *d_group_members,
						const unsigned int *d_nneigh, 
						unsigned int *d_nlist, 
						const unsigned int *d_headlist,
						const float ndsr,
						const float k_n,
						const float kappa,
						const float beta,
						const float epsq,
						const BoxDim box
						);


__global__ void Stokes_SetVelocity_kernel(
						Scalar4 *d_vel,
						float   *d_Velocity,
						unsigned int group_size,
						unsigned int *d_group_members
						);


#endif
