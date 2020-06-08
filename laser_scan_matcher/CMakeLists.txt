PROJECT(laser_scan_matcher)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)


FIND_PACKAGE(catkin REQUIRED COMPONENTS message_generation std_msgs sensor_msgs rospy)

add_service_files(
    DIRECTORY srv
    FILES MatchLaserScans.srv
)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)
catkin_package(CATKIN_DEPENDS std_msgs sensor_msgs message_runtime)
catkin_install_python(PROGRAMS scripts/match_laser_scans.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
