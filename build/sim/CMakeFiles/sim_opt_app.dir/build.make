# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alexie/Documents/slam_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alexie/Documents/slam_demo/build

# Include any dependencies generated for this target.
include sim/CMakeFiles/sim_opt_app.dir/depend.make

# Include the progress variables for this target.
include sim/CMakeFiles/sim_opt_app.dir/progress.make

# Include the compile flags for this target's objects.
include sim/CMakeFiles/sim_opt_app.dir/flags.make

sim/CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.o: sim/CMakeFiles/sim_opt_app.dir/flags.make
sim/CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.o: ../sim/sim_opt_app.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alexie/Documents/slam_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object sim/CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.o"
	cd /home/alexie/Documents/slam_demo/build/sim && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.o -c /home/alexie/Documents/slam_demo/sim/sim_opt_app.cc

sim/CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.i"
	cd /home/alexie/Documents/slam_demo/build/sim && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alexie/Documents/slam_demo/sim/sim_opt_app.cc > CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.i

sim/CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.s"
	cd /home/alexie/Documents/slam_demo/build/sim && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alexie/Documents/slam_demo/sim/sim_opt_app.cc -o CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.s

# Object files for target sim_opt_app
sim_opt_app_OBJECTS = \
"CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.o"

# External object files for target sim_opt_app
sim_opt_app_EXTERNAL_OBJECTS =

sim/sim_opt_app: sim/CMakeFiles/sim_opt_app.dir/sim_opt_app.cc.o
sim/sim_opt_app: sim/CMakeFiles/sim_opt_app.dir/build.make
sim/sim_opt_app: lib/liblib.a
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
sim/sim_opt_app: /usr/local/lib/libceres.a
sim/sim_opt_app: /usr/lib/x86_64-linux-gnu/libglog.so
sim/sim_opt_app: sim/CMakeFiles/sim_opt_app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alexie/Documents/slam_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sim_opt_app"
	cd /home/alexie/Documents/slam_demo/build/sim && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sim_opt_app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sim/CMakeFiles/sim_opt_app.dir/build: sim/sim_opt_app

.PHONY : sim/CMakeFiles/sim_opt_app.dir/build

sim/CMakeFiles/sim_opt_app.dir/clean:
	cd /home/alexie/Documents/slam_demo/build/sim && $(CMAKE_COMMAND) -P CMakeFiles/sim_opt_app.dir/cmake_clean.cmake
.PHONY : sim/CMakeFiles/sim_opt_app.dir/clean

sim/CMakeFiles/sim_opt_app.dir/depend:
	cd /home/alexie/Documents/slam_demo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alexie/Documents/slam_demo /home/alexie/Documents/slam_demo/sim /home/alexie/Documents/slam_demo/build /home/alexie/Documents/slam_demo/build/sim /home/alexie/Documents/slam_demo/build/sim/CMakeFiles/sim_opt_app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sim/CMakeFiles/sim_opt_app.dir/depend

