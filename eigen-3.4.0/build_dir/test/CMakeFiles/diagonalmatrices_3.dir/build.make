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
include test/CMakeFiles/diagonalmatrices_3.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/diagonalmatrices_3.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/diagonalmatrices_3.dir/flags.make

test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o: test/CMakeFiles/diagonalmatrices_3.dir/flags.make
test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o: /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/diagonalmatrices.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o -c /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/diagonalmatrices.cpp

test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.i"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/diagonalmatrices.cpp > CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.i

test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.s"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/diagonalmatrices.cpp -o CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.s

test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o.requires:

.PHONY : test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o.requires

test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o.provides: test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/diagonalmatrices_3.dir/build.make test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o.provides.build
.PHONY : test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o.provides

test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o.provides.build: test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o


# Object files for target diagonalmatrices_3
diagonalmatrices_3_OBJECTS = \
"CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o"

# External object files for target diagonalmatrices_3
diagonalmatrices_3_EXTERNAL_OBJECTS =

test/diagonalmatrices_3: test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o
test/diagonalmatrices_3: test/CMakeFiles/diagonalmatrices_3.dir/build.make
test/diagonalmatrices_3: test/CMakeFiles/diagonalmatrices_3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable diagonalmatrices_3"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/diagonalmatrices_3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/diagonalmatrices_3.dir/build: test/diagonalmatrices_3

.PHONY : test/CMakeFiles/diagonalmatrices_3.dir/build

test/CMakeFiles/diagonalmatrices_3.dir/requires: test/CMakeFiles/diagonalmatrices_3.dir/diagonalmatrices.cpp.o.requires

.PHONY : test/CMakeFiles/diagonalmatrices_3.dir/requires

test/CMakeFiles/diagonalmatrices_3.dir/clean:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && $(CMAKE_COMMAND) -P CMakeFiles/diagonalmatrices_3.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/diagonalmatrices_3.dir/clean

test/CMakeFiles/diagonalmatrices_3.dir/depend:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0 /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test/CMakeFiles/diagonalmatrices_3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/diagonalmatrices_3.dir/depend

