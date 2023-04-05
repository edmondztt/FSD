# Maintainer: Andrew M. Fiore

set(COMPONENT_NAME PSEv3_plugin)

set(_${COMPONENT_NAME}_sources
    module.cc
    Stokes.cc
    Stokes_ResistanceTable.cc
    Stokes_SparseMath.cc
    ShearFunction.cc
    ShearFunctionWrap.cc
    SpecificShearFunction.cc
    VariantShearFunction.cc
    rcm.cpp
    )

set(_${COMPONENT_NAME}_cu_sources
    Brownian_FarField.cu
    Brownian_NearField.cu
    Helper_Brownian.cu
    Helper_Debug.cu
    Helper_Integrator.cu
    Helper_Mobility.cu
    Helper_Precondition.cu
    Helper_Stokes.cu
    Helper_Saddle.cu
    Integrator.cu
    Lubrication.cu
    Mobility.cu
    Precondition.cu
    Saddle.cu
    Stokes.cu
    Solvers.cu
    )

if (ENABLE_HIP)
set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources})
endif (ENABLE_HIP)

pybind11_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_cuda_sources} NO_EXTRAS)
# Alias into the HOOMD namespace so that plugins and symlinked components both work.
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# Find additional libraries to be linked for the plugin
find_library( LAPACKE_LIBRARIES lapacke PATHS /usr/local/lapack-3.6.0/ )
find_library( LAPACK_LIBRARIES lapack PATHS /usr/local/lapack-3.6.0/ )
find_library( BLAS_LIBRARIES blas PATHS /usr/local/lapack-3.6.0/ )
find_library( CBLAS_LIBRARIES cblas PATHS /usr/local/lapack-3.6.0/ )
if( LAPACKE_LIBRARIES-NOTFOUND )
        message(FATAL_ERROR "lapacke libraries not found")
endif( LAPACKE_LIBRARIES-NOTFOUND )
message(STATUS  "found lapacke libraries: ${LAPACKE_LIBRARIES}")
if( LAPACK_LIBRARIES-NOTFOUND )
        message(FATAL_ERROR "lapack libraries not found")
endif( LAPACK_LIBRARIES-NOTFOUND )
message(STATUS  "found lapack libraries: ${LAPACK_LIBRARIES}")
set( LAPACK_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES}  ${BLAS_LIBRARIES} )
target_include_directories(_${COMPONENT_NAME} PUBLIC /usr/local/lapack-3.6.0/)
#include_directories(/usr/local/lapack-3.6.0/)

# Find CUBLAS
find_library( CUBLAS_LIBRARY cublas PATHS /usr/local/cuda/lib64 )
if( CUBLAS_LIBRARY-NOTFOUND )
	message(FATAL_ERROR "CUBLAS Library not found")
endif( CUBLAS_LIBRARY-NOTFOUND )
message(STATUS  "found CUBLAS library: ${CUBLAS_LIBRARY}")
set( CUBLAS_LIBRARY ${CUBLAS_LIBRARY} )

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd PRIVATE ${LAPACK_LIBRARIES} ${CUBLAS_LIBRARY})

# Edmond 04/04/2023: comment out this part to debug -pthread repeats
# # if we are compiling with MPI support built in, set appropriate
# # compiler/linker flags
# if (ENABLE_MPI)
#    if(MPI_COMPILE_FLAGS)
#        set_target_properties(_${COMPONENT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
#    endif(MPI_COMPILE_FLAGS)
#    if(MPI_LINK_FLAGS)
#        set_target_properties(_${COMPONENT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
#    endif(MPI_LINK_FLAGS)
# endif(ENABLE_MPI)

fix_cudart_rpath(_${COMPONENT_NAME})

# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# # copy python modules to the build directory to make it a working python package
# MACRO(copy_file file)
#     add_custom_command (
#         OUTPUT ${file}
#         DEPENDS ${file}
#         POST_BUILD
#         COMMAND    ${CMAKE_COMMAND}
#         ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
#         COMMENT    "Copy hoomd/${COMPONENT_NAME}/${file}"
#     )
# ENDMACRO(copy_file)

set(files
    __init__.py
    integrate.py
    shear_function.py
    variant.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

# foreach(file ${files})
#     copy_file(${file})
# endforeach()

copy_files_to_build("${files}" "${COMPONENT_NAME}" "*.py")

