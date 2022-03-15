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
include doc/examples/CMakeFiles/nullary_indexing.dir/depend.make

# Include the progress variables for this target.
include doc/examples/CMakeFiles/nullary_indexing.dir/progress.make

# Include the compile flags for this target's objects.
include doc/examples/CMakeFiles/nullary_indexing.dir/flags.make

doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o: doc/examples/CMakeFiles/nullary_indexing.dir/flags.make
doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o: /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples/nullary_indexing.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o -c /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples/nullary_indexing.cpp

doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.i"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples/nullary_indexing.cpp > CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.i

doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.s"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples/nullary_indexing.cpp -o CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.s

doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o.requires:

.PHONY : doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o.requires

doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o.provides: doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o.requires
	$(MAKE) -f doc/examples/CMakeFiles/nullary_indexing.dir/build.make doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o.provides.build
.PHONY : doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o.provides

doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o.provides.build: doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o


# Object files for target nullary_indexing
nullary_indexing_OBJECTS = \
"CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o"

# External object files for target nullary_indexing
nullary_indexing_EXTERNAL_OBJECTS =

doc/examples/nullary_indexing: doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o
doc/examples/nullary_indexing: doc/examples/CMakeFiles/nullary_indexing.dir/build.make
doc/examples/nullary_indexing: doc/examples/CMakeFiles/nullary_indexing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable nullary_indexing"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nullary_indexing.dir/link.txt --verbose=$(VERBOSE)
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && ./nullary_indexing >/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples/nullary_indexing.out

# Rule to build all files generated by this target.
doc/examples/CMakeFiles/nullary_indexing.dir/build: doc/examples/nullary_indexing

.PHONY : doc/examples/CMakeFiles/nullary_indexing.dir/build

doc/examples/CMakeFiles/nullary_indexing.dir/requires: doc/examples/CMakeFiles/nullary_indexing.dir/nullary_indexing.cpp.o.requires

.PHONY : doc/examples/CMakeFiles/nullary_indexing.dir/requires

doc/examples/CMakeFiles/nullary_indexing.dir/clean:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/nullary_indexing.dir/cmake_clean.cmake
.PHONY : doc/examples/CMakeFiles/nullary_indexing.dir/clean

doc/examples/CMakeFiles/nullary_indexing.dir/depend:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0 /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples/CMakeFiles/nullary_indexing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/examples/CMakeFiles/nullary_indexing.dir/depend

