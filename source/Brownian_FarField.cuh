// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Brownian_FarField.cuh
    \brief Declares GPU kernel code for far-field Brownian Calculation.
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

#include "DataStruct.h"

//! Define the kernel
#ifndef __BROWNIAN_FARFIELD_CUH__
#define __BROWNIAN_FARFIELD_CUH__

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
void Brownian_FarField_SlipVelocity(
			        	Scalar *d_Uslip_ff,
					Scalar4 *d_pos,
                                	unsigned int *d_group_members,
                                	unsigned int group_size,
                                	const BoxDim& box,
                                	Scalar dt,
			        	BrownianData *bro_data,
			        	MobilityData *mob_data,
					KernelData *ker_data,
					WorkData *work_data
					);
}
#endif
