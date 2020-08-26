PROJECT(local_uncertainty_estimator)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

MESSAGE(STATUS "Building Local Uncertainty Estimator")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

FIND_PACKAGE(catkin REQUIRED COMPONENTS roslib std_msgs sensor_msgs message_generation)

add_service_files(
    DIRECTORY srv
    FILES
    EstimateLocalUncertainty.srv
)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)
