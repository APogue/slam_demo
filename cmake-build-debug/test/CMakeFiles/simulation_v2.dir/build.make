# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

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

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2020.3.1/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/clion-2020.3.1/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alexie/Documents/slam_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alexie/Documents/slam_demo/cmake-build-debug

# Include any dependencies generated for this target.
include test/CMakeFiles/simulation_v2.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/simulation_v2.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/simulation_v2.dir/flags.make

test/CMakeFiles/simulation_v2.dir/simulation_v2.cc.o: test/CMakeFiles/simulation_v2.dir/flags.make
test/CMakeFiles/simulation_v2.dir/simulation_v2.cc.o: ../test/simulation_v2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alexie/Documents/slam_demo/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/simulation_v2.dir/simulation_v2.cc.o"
	cd /home/alexie/Documents/slam_demo/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simulation_v2.dir/simulation_v2.cc.o -c /home/alexie/Documents/slam_demo/test/simulation_v2.cc

test/CMakeFiles/simulation_v2.dir/simulation_v2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simulation_v2.dir/simulation_v2.cc.i"
	cd /home/alexie/Documents/slam_demo/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alexie/Documents/slam_demo/test/simulation_v2.cc > CMakeFiles/simulation_v2.dir/simulation_v2.cc.i

test/CMakeFiles/simulation_v2.dir/simulation_v2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simulation_v2.dir/simulation_v2.cc.s"
	cd /home/alexie/Documents/slam_demo/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alexie/Documents/slam_demo/test/simulation_v2.cc -o CMakeFiles/simulation_v2.dir/simulation_v2.cc.s

# Object files for target simulation_v2
simulation_v2_OBJECTS = \
"CMakeFiles/simulation_v2.dir/simulation_v2.cc.o"

# External object files for target simulation_v2
simulation_v2_EXTERNAL_OBJECTS =

test/simulation_v2: test/CMakeFiles/simulation_v2.dir/simulation_v2.cc.o
test/simulation_v2: test/CMakeFiles/simulation_v2.dir/build.make
test/simulation_v2: lib/liblib.a
test/simulation_v2: /usr/local/lib/libceres.a
test/simulation_v2: /usr/lib/x86_64-linux-gnu/libglog.so
test/simulation_v2: test/CMakeFiles/simulation_v2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alexie/Documents/slam_demo/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable simulation_v2"
	cd /home/alexie/Documents/slam_demo/cmake-build-debug/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simulation_v2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/simulation_v2.dir/build: test/simulation_v2

.PHONY : test/CMakeFiles/simulation_v2.dir/build

test/CMakeFiles/simulation_v2.dir/clean:
	cd /home/alexie/Documents/slam_demo/cmake-build-debug/test && $(CMAKE_COMMAND) -P CMakeFiles/simulation_v2.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/simulation_v2.dir/clean

test/CMakeFiles/simulation_v2.dir/depend:
	cd /home/alexie/Documents/slam_demo/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alexie/Documents/slam_demo /home/alexie/Documents/slam_demo/test /home/alexie/Documents/slam_demo/cmake-build-debug /home/alexie/Documents/slam_demo/cmake-build-debug/test /home/alexie/Documents/slam_demo/cmake-build-debug/test/CMakeFiles/simulation_v2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/simulation_v2.dir/depend

