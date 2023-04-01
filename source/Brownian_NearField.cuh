// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Brownian_NearField.cuh
    \brief Declares GPU kernel code for Near-Field Brownian Calculation
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

#include "DataStruct.h"

#include <cusparse.h>
#include <cusolverSp.h>

//! Define the kernel
#ifndef __BROWNIAN_NEARFIELD_CUH__
#define __BROWNIAN_NEARFIELD_CUH__

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

__global__ void Brownian_NearField_RNG_kernel(
						Scalar *d_Psi_nf,
						unsigned int N,
						const unsigned int seed,
						const Scalar T,
						const Scalar dt
						);


void Brownian_NearField_Force(
				Scalar *d_FBnf, // output
				Scalar4 *d_pos,
				unsigned int *d_group_members,
                                unsigned int group_size,
                                const BoxDim& box,
                                Scalar dt,
				void *pBuffer,
				KernelData *ker_data,
				BrownianData *bro_data,
				ResistanceData *res_data,
				WorkData *work_data
				);

#endif
