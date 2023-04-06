// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Saddle.cuh
    \brief Declared functions for saddle point calculations
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

#include <stdlib.h>

#include <cusparse.h>
#include <cusolverSp.h>

//! Define the step_one kernel
#ifndef __SADDLE_CUH__
#define __SADDLE_CUH__

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

void Saddle_Multiply( 
                        	Scalar *d_b, // output
				Scalar *d_x, // input
				Scalar4 *d_pos,
				unsigned int *d_group_members,
				unsigned int group_size,
                        	const BoxDim& box,
				KernelData *ker_data,
				MobilityData *mob_data,
				ResistanceData *res_data,
				WorkData *work_data
				);

void Saddle_Preconditioner(	
				Scalar *d_x, // Solution
				Scalar *d_b, // RHS
				int group_size,
				void *pBuffer,
				KernelData *ker_data,
				ResistanceData *res_data
				);

}
#endif
