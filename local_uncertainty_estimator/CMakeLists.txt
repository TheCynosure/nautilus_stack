PROJECT(local_uncertainty_estimator)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)


FIND_PACKAGE(catkin REQUIRED COMPONENTS message_generation std_msgs sensor_msgs rospy)

add_service_files(
    DIRECTORY srv
    FILES EstimateLocalUncertainty.srv
)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)
catkin_package(CATKIN_DEPENDS std_msgs sensor_msgs message_runtime)
catkin_install_python(PROGRAMS scripts/estimate_local_uncertainty.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)