// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Integrator.cuh
    \brief Declares helper functions for integration.
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

//! Define the step_one kernel
#ifndef __HELPER_INTEGRATOR_CUH__
#define __HELPER_INTEGRATOR_CUH__

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

__global__ void Integrator_RFD_RandDisp_kernel(
						Scalar *d_psi,
						unsigned int N,
						const unsigned int seed
						);
__global__ void Integrator_ZeroVelocity_kernel( 
						Scalar *d_b,
						unsigned int N
						);
__global__ void Integrator_AddStrainRate_kernel( 
						Scalar *d_b,
						Scalar shear_rate,
						unsigned int N
						);
}

#endif
