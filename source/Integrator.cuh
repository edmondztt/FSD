// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Zhouyang Ge

/*! \file Integrator.cuh
    \brief Declares GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.
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
#ifndef __INTEGRATOR_CUH__
#define __INTEGRATOR_CUH__

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

extern "C" __global__ void Integrator_ExplicitEuler_kernel(	
								Scalar4 *d_pos_in,
								Scalar4 *d_pos_out,
                             					Scalar *d_Velocity,
                             					int3 *d_image,
                             					unsigned int *d_group_members,
                             					unsigned int group_size,
                             					BoxDim box,
                             					Scalar dt 
								);

extern "C" __global__ void Integrator_ExplicitEuler_Shear_kernel(
								 Scalar4 *d_pos_in,
								 Scalar4 *d_pos_out,
								 Scalar *d_Velocity,
								 int3 *d_image,
								 unsigned int *d_group_members,
								 unsigned int group_size,
								 BoxDim box,
								 Scalar dt,
								 Scalar shear_rate
								 );


extern "C" __global__ void Integrator_RK_Shear_kernel(Scalar coef_1, Scalar4 *d_pos_in_1,
						      Scalar coef_2, Scalar4 *d_pos_in_2,
						      Scalar4 *d_pos_out,
						      Scalar *d_Velocity,
						      int3 *d_image,
						      unsigned int *d_group_members,
						      unsigned int group_size,
						      BoxDim box,
						      Scalar coef_3, Scalar dt,
						      Scalar shear_rate
						      );


extern "C" __global__ void Integrator_buffer_vel_kernel(Scalar *d_Velocity,
							Scalar3 *vel_rk,
							unsigned int *d_group_members,
							unsigned int group_size
							);

extern "C" __global__ void Integrator_supim_kernel(Scalar3 *vel_rk1,  
						   Scalar3 *vel_rk2,  
						   Scalar3 *vel_rk3,  
						   Scalar3 *vel_rk4,
						   Scalar *d_Velocity,
						   unsigned int *d_group_members,
						   unsigned int group_size
						   );


extern "C" __global__ void Integrator_AB2_Shear_kernel(
						       unsigned int timestep, Scalar4 *d_vel,
								Scalar4 *d_pos_in,
								Scalar4 *d_pos_out,
                             					Scalar *d_Velocity,
                             					int3 *d_image,
                             					unsigned int *d_group_members,
                             					unsigned int group_size,
                             					BoxDim box,
                             					Scalar dt,
								Scalar shear_rate
								);

void Integrator_RFD(
			Scalar *d_Divergence, // 11*N (will have some zeros, but they will be ignored later)
			Scalar4 *d_pos,
			int3 *d_image,
			unsigned int *d_group_members,
			unsigned int group_size,
			const BoxDim& box,
			KernelData *ker_data,
			BrownianData *bro_data,
			MobilityData *mob_data,
			ResistanceData *res_data,
			WorkData *work_data
			);

void Integrator_ComputeVelocity(     unsigned int timestep,
				     unsigned int output_period,
			
					Scalar *d_AppliedForce,
					Scalar *d_Velocity,
					Scalar dt,
					Scalar shear_rate,
					Scalar4 *d_pos,
					int3 *d_image,
					unsigned int *d_group_members,
					unsigned int group_size,
					const BoxDim& box,
					KernelData *ker_data,
					BrownianData *bro_data,
					MobilityData *mob_data,
					ResistanceData *res_data,
					WorkData *work_data
					);

}

#endif
