// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Saddle.cuh
    \brief Declared helper functions for saddle point calculations
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

#include <stdlib.h>
#include "cusparse.h"

//! Define the step_one kernel
#ifndef __HELPER_SADDLE_CUH__
#define __HELPER_SADDLE_CUH__

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

__global__ void Saddle_ZeroOutput_kernel( 
						Scalar *d_b, 
						unsigned int N 
						);

__global__ void Saddle_AddFloat_kernel( 	Scalar *d_a, 
						Scalar *d_b,
						Scalar *d_c,
						Scalar coeff_a,
						Scalar coeff_b,
						unsigned int N,
						int stride
					);

__global__ void Saddle_SplitGeneralizedF_kernel( 	Scalar *d_GeneralF, 
							Scalar4 *d_net_force,
							Scalar4 *d_TorqueStress,
							unsigned int N
					);

__global__ void Saddle_MakeGeneralizedU_kernel( 	Scalar *d_GeneralU, 
							Scalar4 *d_vel,
							Scalar4 *d_AngvelStrain,
							unsigned int N
					);


__global__ void Saddle_force2rhs_kernel(
					Scalar *d_force, 
					Scalar *d_rhs,
					unsigned int N
					);

__global__ void Saddle_solution2vel_kernel(
					Scalar *d_U, 
					Scalar *d_solution,
					unsigned int N
					);

}

#endif
