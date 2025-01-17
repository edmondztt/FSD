// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Saddle_Helper.cuh
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

#include <stdlib.h>

#include "DataStruct.h"

#include <cusparse.h>
#include <cusolverSp.h>

//! Define the step_one kernel
#ifndef __SOLVERS_CUH__
#define __SOLVERS_CUH__

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

void Solvers_Saddle(
			Scalar *d_rhs, 
			Scalar *d_solution,
			Scalar4 *d_pos,
			unsigned int *d_group_members,
			unsigned int group_size,
			const BoxDim& box,
			Scalar tol,
			void *pBuffer,
			KernelData *ker_data,
			MobilityData *mob_data,
			ResistanceData *res_data,
			WorkData *work_data
			);

}

#endif
