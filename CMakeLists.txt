cmake_minimum_required(VERSION 2.8)

#setOpenCV_DIR
#set(OpenCV_DIR "/home/andyoyo/opencv/opencv-master/build")

<<<<<<< HEAD
project(bankSegment)
=======
project(map_PS)
>>>>>>> 84a8aab78b5134eeea266d6a095b845b151e0c85

find_package( OpenCV REQUIRED)

#print OpenCV_VERSION on terminal

message(STATUS "OpenCV --version: " ${OpenCV_VERSION})

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")

<<<<<<< HEAD
add_executable(bankSegment ${sourcefiles} )
target_link_libraries(bankSegment ${OpenCV_LIBS})
=======
add_executable(map_PS ${sourcefiles} )
target_link_libraries(map_PS ${OpenCV_LIBS})
>>>>>>> 84a8aab78b5134eeea266d6a095b845b151e0c85
