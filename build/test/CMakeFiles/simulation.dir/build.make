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
include test/CMakeFiles/simulation.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/simulation.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/simulation.dir/flags.make

test/CMakeFiles/simulation.dir/simulation.cc.o: test/CMakeFiles/simulation.dir/flags.make
test/CMakeFiles/simulation.dir/simulation.cc.o: ../test/simulation.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alexie/Documents/slam_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/simulation.dir/simulation.cc.o"
	cd /home/alexie/Documents/slam_demo/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simulation.dir/simulation.cc.o -c /home/alexie/Documents/slam_demo/test/simulation.cc

test/CMakeFiles/simulation.dir/simulation.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simulation.dir/simulation.cc.i"
	cd /home/alexie/Documents/slam_demo/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alexie/Documents/slam_demo/test/simulation.cc > CMakeFiles/simulation.dir/simulation.cc.i

test/CMakeFiles/simulation.dir/simulation.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simulation.dir/simulation.cc.s"
	cd /home/alexie/Documents/slam_demo/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alexie/Documents/slam_demo/test/simulation.cc -o CMakeFiles/simulation.dir/simulation.cc.s

# Object files for target simulation
simulation_OBJECTS = \
"CMakeFiles/simulation.dir/simulation.cc.o"

# External object files for target simulation
simulation_EXTERNAL_OBJECTS =

test/simulation: test/CMakeFiles/simulation.dir/simulation.cc.o
test/simulation: test/CMakeFiles/simulation.dir/build.make
test/simulation: lib/liblib.a
test/simulation: /usr/local/lib/libceres.a
test/simulation: /usr/lib/x86_64-linux-gnu/libglog.so
test/simulation: test/CMakeFiles/simulation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alexie/Documents/slam_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable simulation"
	cd /home/alexie/Documents/slam_demo/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simulation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/simulation.dir/build: test/simulation

.PHONY : test/CMakeFiles/simulation.dir/build

test/CMakeFiles/simulation.dir/clean:
	cd /home/alexie/Documents/slam_demo/build/test && $(CMAKE_COMMAND) -P CMakeFiles/simulation.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/simulation.dir/clean

test/CMakeFiles/simulation.dir/depend:
	cd /home/alexie/Documents/slam_demo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alexie/Documents/slam_demo /home/alexie/Documents/slam_demo/test /home/alexie/Documents/slam_demo/build /home/alexie/Documents/slam_demo/build/test /home/alexie/Documents/slam_demo/build/test/CMakeFiles/simulation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/simulation.dir/depend

