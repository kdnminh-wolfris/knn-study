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
include test/CMakeFiles/blasutil_8.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/blasutil_8.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/blasutil_8.dir/flags.make

test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o: test/CMakeFiles/blasutil_8.dir/flags.make
test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o: /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/blasutil.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/blasutil_8.dir/blasutil.cpp.o -c /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/blasutil.cpp

test/CMakeFiles/blasutil_8.dir/blasutil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/blasutil_8.dir/blasutil.cpp.i"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/blasutil.cpp > CMakeFiles/blasutil_8.dir/blasutil.cpp.i

test/CMakeFiles/blasutil_8.dir/blasutil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/blasutil_8.dir/blasutil.cpp.s"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/blasutil.cpp -o CMakeFiles/blasutil_8.dir/blasutil.cpp.s

test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o.requires:

.PHONY : test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o.requires

test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o.provides: test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/blasutil_8.dir/build.make test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o.provides.build
.PHONY : test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o.provides

test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o.provides.build: test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o


# Object files for target blasutil_8
blasutil_8_OBJECTS = \
"CMakeFiles/blasutil_8.dir/blasutil.cpp.o"

# External object files for target blasutil_8
blasutil_8_EXTERNAL_OBJECTS =

test/blasutil_8: test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o
test/blasutil_8: test/CMakeFiles/blasutil_8.dir/build.make
test/blasutil_8: test/CMakeFiles/blasutil_8.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable blasutil_8"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/blasutil_8.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/blasutil_8.dir/build: test/blasutil_8

.PHONY : test/CMakeFiles/blasutil_8.dir/build

test/CMakeFiles/blasutil_8.dir/requires: test/CMakeFiles/blasutil_8.dir/blasutil.cpp.o.requires

.PHONY : test/CMakeFiles/blasutil_8.dir/requires

test/CMakeFiles/blasutil_8.dir/clean:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && $(CMAKE_COMMAND) -P CMakeFiles/blasutil_8.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/blasutil_8.dir/clean

test/CMakeFiles/blasutil_8.dir/depend:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0 /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test/CMakeFiles/blasutil_8.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/blasutil_8.dir/depend

