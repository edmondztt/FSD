// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Preconditioner.cuh
    \brief Define the GPU kernels and driving functions to compute the preconditioner. 
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
#ifndef __PRECONDITION_CUH__
#define __PRECONDITION_CUH__

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

void Precondition_Brownian_RFUmultiply(	
					Scalar *d_y,       // output
					Scalar *d_x,       // input
					const Scalar4 *d_pos,
					unsigned int *d_group_members,
					const int group_size, 
			      		const BoxDim box,
					void *pBuffer,
					KernelData *ker_data,
					ResistanceData *res_data
					);

void Precondition_Brownian_Undo(	
				Scalar *d_x,       // input/output
				int group_size,
				KernelData *ker_data,
				ResistanceData *res_data
				);

void Precondition_Saddle_RFUmultiply(	
					Scalar *d_y,       // output
					Scalar *d_x,       // input
					Scalar *d_Scratch, // intermediate storage
					const int *d_prcm,
					int group_size,
					unsigned int nnz,
					const int   *d_L_RowPtr,
					const int   *d_L_ColInd,
					const Scalar *d_L_Val,
					cusparseHandle_t spHandle,
        				cusparseStatus_t spStatus,
					cusparseMatDescr_t descr_L,
					csrsv2Info_t info_L,
					csrsv2Info_t info_Lt,
					const cusparseOperation_t trans_L,
					const cusparseOperation_t trans_Lt,
					const cusparseSolvePolicy_t policy_L,
					const cusparseSolvePolicy_t policy_Lt,
					void *pBuffer,
					dim3 grid,
					dim3 threads
					);

void Precondition_Wrap(
			Scalar4 *d_pos,
			unsigned int *d_group_members,
			unsigned int group_size,
			const BoxDim& box,
			KernelData *ker_data,
			ResistanceData *res_data,
			WorkData *work_data
			);

}

#endif
