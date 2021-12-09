cmake_minimum_required(VERSION 3.0.2)
project( DisplayImage )

find_package( OpenCV 4 REQUIRED )

find_package (Eigen3 3.3 REQUIRED)
include_directories( EIGEN3_INCLUDE_DIR )

find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
include_directories( 
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
     )



add_executable( sort
src/sort.cpp 
src/include/run_yolo.cpp
src/include/hungarian.cpp
)
target_link_libraries( sort
 ${OpenCV_LIBS} 
 )

