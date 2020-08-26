PROJECT(point_cloud_embedder)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

MESSAGE(STATUS "Building Point Cloud Embedder")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

INCLUDE($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

SET(ROS_BUILD_STATIC_LIBS true)
SET(ROS_BUILD_SHARED_LIBS false)

MESSAGE(STATUS "ROS-Overrride Build Type: ${CMAKE_BUILD_TYPE}")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_service_files(
    DIRECTORY srv
    FILES
    GetPointCloudEmbedding.srv
)

rosbuild_gensrv()

SET(libs roslib)
