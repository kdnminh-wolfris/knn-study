# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir

# Include any dependencies generated for this target.
include failtest/CMakeFiles/ref_4_ko.dir/depend.make

# Include the progress variables for this target.
include failtest/CMakeFiles/ref_4_ko.dir/progress.make

# Include the compile flags for this target's objects.
include failtest/CMakeFiles/ref_4_ko.dir/flags.make

failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o: failtest/CMakeFiles/ref_4_ko.dir/flags.make
failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o: /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/failtest/ref_4.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/failtest && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ref_4_ko.dir/ref_4.cpp.o -c /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/failtest/ref_4.cpp

failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ref_4_ko.dir/ref_4.cpp.i"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/failtest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/failtest/ref_4.cpp > CMakeFiles/ref_4_ko.dir/ref_4.cpp.i

failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ref_4_ko.dir/ref_4.cpp.s"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/failtest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/failtest/ref_4.cpp -o CMakeFiles/ref_4_ko.dir/ref_4.cpp.s

failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o.requires:

.PHONY : failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o.requires

failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o.provides: failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o.requires
	$(MAKE) -f failtest/CMakeFiles/ref_4_ko.dir/build.make failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o.provides.build
.PHONY : failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o.provides

failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o.provides.build: failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o


# Object files for target ref_4_ko
ref_4_ko_OBJECTS = \
"CMakeFiles/ref_4_ko.dir/ref_4.cpp.o"

# External object files for target ref_4_ko
ref_4_ko_EXTERNAL_OBJECTS =

failtest/ref_4_ko: failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o
failtest/ref_4_ko: failtest/CMakeFiles/ref_4_ko.dir/build.make
failtest/ref_4_ko: failtest/CMakeFiles/ref_4_ko.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ref_4_ko"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/failtest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ref_4_ko.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
failtest/CMakeFiles/ref_4_ko.dir/build: failtest/ref_4_ko

.PHONY : failtest/CMakeFiles/ref_4_ko.dir/build

failtest/CMakeFiles/ref_4_ko.dir/requires: failtest/CMakeFiles/ref_4_ko.dir/ref_4.cpp.o.requires

.PHONY : failtest/CMakeFiles/ref_4_ko.dir/requires

failtest/CMakeFiles/ref_4_ko.dir/clean:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/failtest && $(CMAKE_COMMAND) -P CMakeFiles/ref_4_ko.dir/cmake_clean.cmake
.PHONY : failtest/CMakeFiles/ref_4_ko.dir/clean

failtest/CMakeFiles/ref_4_ko.dir/depend:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0 /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/failtest /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/failtest /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/failtest/CMakeFiles/ref_4_ko.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : failtest/CMakeFiles/ref_4_ko.dir/depend

