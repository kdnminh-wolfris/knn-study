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
include test/CMakeFiles/jacobi_5.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/jacobi_5.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/jacobi_5.dir/flags.make

test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o: test/CMakeFiles/jacobi_5.dir/flags.make
test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o: /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/jacobi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jacobi_5.dir/jacobi.cpp.o -c /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/jacobi.cpp

test/CMakeFiles/jacobi_5.dir/jacobi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jacobi_5.dir/jacobi.cpp.i"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/jacobi.cpp > CMakeFiles/jacobi_5.dir/jacobi.cpp.i

test/CMakeFiles/jacobi_5.dir/jacobi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jacobi_5.dir/jacobi.cpp.s"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/jacobi.cpp -o CMakeFiles/jacobi_5.dir/jacobi.cpp.s

test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o.requires:

.PHONY : test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o.requires

test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o.provides: test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/jacobi_5.dir/build.make test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o.provides.build
.PHONY : test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o.provides

test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o.provides.build: test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o


# Object files for target jacobi_5
jacobi_5_OBJECTS = \
"CMakeFiles/jacobi_5.dir/jacobi.cpp.o"

# External object files for target jacobi_5
jacobi_5_EXTERNAL_OBJECTS =

test/jacobi_5: test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o
test/jacobi_5: test/CMakeFiles/jacobi_5.dir/build.make
test/jacobi_5: test/CMakeFiles/jacobi_5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable jacobi_5"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jacobi_5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/jacobi_5.dir/build: test/jacobi_5

.PHONY : test/CMakeFiles/jacobi_5.dir/build

test/CMakeFiles/jacobi_5.dir/requires: test/CMakeFiles/jacobi_5.dir/jacobi.cpp.o.requires

.PHONY : test/CMakeFiles/jacobi_5.dir/requires

test/CMakeFiles/jacobi_5.dir/clean:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && $(CMAKE_COMMAND) -P CMakeFiles/jacobi_5.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/jacobi_5.dir/clean

test/CMakeFiles/jacobi_5.dir/depend:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0 /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test/CMakeFiles/jacobi_5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/jacobi_5.dir/depend

