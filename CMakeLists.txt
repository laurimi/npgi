cmake_minimum_required(VERSION 3.0)
project(NPGI)

if(BUILD_TEST)
	include(CTest)

	if (CMAKE_VERSION VERSION_LESS 3.2)
	    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
	else()
	    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
	endif()

	include(DownloadProject.cmake)
	download_project(PROJ                googletest
	                 GIT_REPOSITORY      https://github.com/google/googletest.git
	                 GIT_TAG             master
	                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
	)

	# Prevent GoogleTest from overriding our compiler/linker options
	# when building with Visual Studio
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

	add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})


	# When using CMake 2.8.11 or later, header path dependencies
	# are automatically added to the gtest and gmock targets.
	# For earlier CMake versions, we have to explicitly add the
	# required directories to the header search path ourselves.
	if (CMAKE_VERSION VERSION_LESS 2.8.11)
	    include_directories("${gtest_SOURCE_DIR}/include"
	                        "${gmock_SOURCE_DIR}/include")
	endif()
endif()


## MADP toolbox
set(EXT_DIR "${CMAKE_CURRENT_BINARY_DIR}/third-party/ExternalProject")
message(STATUS "EXT_DIR: ${EXT_DIR}")

if(MADPPATH)
	set(MADP_INCLUDE_DIRECTORIES ${MADPPATH}/include)
	set(MADP_LIBRARY_PATH ${MADPPATH}/lib)
else()
	include(ExternalProject)
	ExternalProject_Add(project_madp
		GIT_REPOSITORY	   https://github.com/MADPToolbox/MADP
		GIT_TAG			   a6c1bb7ca18236fc7214d05c6c08dd7d515c4e96
	    PREFIX            ${EXT_DIR}/madp
	    BUILD_IN_SOURCE	  1
	    CONFIGURE_COMMAND sh <SOURCE_DIR>/autogen.sh && <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
	    BUILD_COMMAND make
	    INSTALL_COMMAND make install
	)

	ExternalProject_Get_Property(project_madp INSTALL_DIR BINARY_DIR SOURCE_DIR)
    set(MADP_INCLUDE_DIRECTORIES ${INSTALL_DIR}/include ${SOURCE_DIR}/src/include)
	set(MADP_LIBRARY_PATH ${INSTALL_DIR}/lib)

	message(STATUS "MADP_INCLUDE_DIRECTORIES: ${MADP_INCLUDE_DIRECTORIES}")
	message(STATUS "MADP_LIBRARY_PATH: ${MADP_LIBRARY_PATH}")
	add_library(MADP STATIC IMPORTED)
	set_property(TARGET MADP PROPERTY IMPORTED_LOCATION ${MADP_LIBRARY_PATH}/libMADP-0.4.1.so)

	add_dependencies(MADP project_madp)
	
endif()

include_directories(include 
					${EIGEN3_INCLUDE_DIR}
					SYSTEM ${MADP_INCLUDE_DIRECTORIES})
link_directories(${MADP_LIBRARY_PATH})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX14)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

add_definitions(
    -Wfatal-errors
    )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

FIND_PACKAGE(Boost 1.58 COMPONENTS program_options regex graph REQUIRED)

find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

# Add warnings and errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")




add_library(MADPWrapper src/MADPWrapper.cpp)
target_link_libraries(MADPWrapper MADP Threads::Threads)

add_library(Base
					src/BeliefUtilities.cpp
					src/DecPOMDPUtilities.cpp
					src/History.cpp
					src/HistoryData.cpp
					src/HistoryCacheUtils.cpp
					src/PolicyGraph.cpp 
					src/LocalPolicyHistory.cpp
					src/JointPolicy.cpp
					src/JointPolicyHistory.cpp
					src/JointPolicyHistories.cpp
					src/JointPolicyUtilities.cpp
					src/PolicyInitialization.cpp
					src/PolicyGraphTraversal.cpp
					src/MADPWrapperUtils.cpp
					src/ValueFunction.cpp
)

target_link_libraries(Base MADPWrapper)

add_library(NPGIPlanner
					src/BackwardPass.cpp
					src/ValueUtilities.cpp
					
					)

target_link_libraries(NPGIPlanner Base)


add_library(NPGIParticles
				src/ParticleUtilities.cpp)

add_executable(${PROJECT_NAME} 
				src/main.cpp
				)

add_executable(solve_graphsensing
				src/solve_graphsensing.cpp
				src/GraphSensingProblem.cpp
				)

target_link_libraries(${PROJECT_NAME} MADPWrapper Base NPGIPlanner ${Boost_LIBRARIES})
target_link_libraries(solve_graphsensing Base NPGIParticles ${Boost_LIBRARIES})


if (BUILD_TEST)
	add_executable(test_graph test/test_graph.cpp)
	target_link_libraries(test_graph Base gtest gtest_main ${Boost_LIBRARIES})
	add_test(NAME test_graph COMMAND test_graph)

	add_executable(test_beliefsparse test/test_beliefsparse.cpp)
	target_link_libraries(test_beliefsparse Base gtest gtest_main)
	add_test(NAME test_beliefsparse COMMAND test_beliefsparse)

	add_executable(test_observationmatrix test/test_observationmatrix.cpp)
	target_link_libraries(test_observationmatrix Base gtest gtest_main)
	add_test(NAME test_observationmatrix COMMAND test_observationmatrix)

	add_executable(test_statetransitionmatrix test/test_statetransitionmatrix.cpp)
	target_link_libraries(test_statetransitionmatrix Base gtest gtest_main)
	add_test(NAME test_statetransitionmatrix COMMAND test_statetransitionmatrix)

	add_executable(test_eigenutils test/test_eigenutils.cpp)
	target_link_libraries(test_eigenutils gtest gtest_main)
	add_test(NAME test_eigenutils COMMAND test_eigenutils)

	add_executable(test_graphsensingproblem test/test_GraphSensingModel.cpp src/GraphSensingProblem.cpp src/ParticleUtilities.cpp)
	target_link_libraries(test_graphsensingproblem Base NPGIPlanner gtest gtest_main)
	add_test(NAME test_graphsensingproblem COMMAND test_graphsensingproblem)
endif()
