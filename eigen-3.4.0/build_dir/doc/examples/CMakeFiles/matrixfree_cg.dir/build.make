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
include doc/examples/CMakeFiles/matrixfree_cg.dir/depend.make

# Include the progress variables for this target.
include doc/examples/CMakeFiles/matrixfree_cg.dir/progress.make

# Include the compile flags for this target's objects.
include doc/examples/CMakeFiles/matrixfree_cg.dir/flags.make

doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o: doc/examples/CMakeFiles/matrixfree_cg.dir/flags.make
doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o: /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples/matrixfree_cg.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o -c /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples/matrixfree_cg.cpp

doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.i"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples/matrixfree_cg.cpp > CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.i

doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.s"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples/matrixfree_cg.cpp -o CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.s

doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o.requires:

.PHONY : doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o.requires

doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o.provides: doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o.requires
	$(MAKE) -f doc/examples/CMakeFiles/matrixfree_cg.dir/build.make doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o.provides.build
.PHONY : doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o.provides

doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o.provides.build: doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o


# Object files for target matrixfree_cg
matrixfree_cg_OBJECTS = \
"CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o"

# External object files for target matrixfree_cg
matrixfree_cg_EXTERNAL_OBJECTS =

doc/examples/matrixfree_cg: doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o
doc/examples/matrixfree_cg: doc/examples/CMakeFiles/matrixfree_cg.dir/build.make
doc/examples/matrixfree_cg: doc/examples/CMakeFiles/matrixfree_cg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable matrixfree_cg"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrixfree_cg.dir/link.txt --verbose=$(VERBOSE)
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && ./matrixfree_cg >/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples/matrixfree_cg.out

# Rule to build all files generated by this target.
doc/examples/CMakeFiles/matrixfree_cg.dir/build: doc/examples/matrixfree_cg

.PHONY : doc/examples/CMakeFiles/matrixfree_cg.dir/build

doc/examples/CMakeFiles/matrixfree_cg.dir/requires: doc/examples/CMakeFiles/matrixfree_cg.dir/matrixfree_cg.cpp.o.requires

.PHONY : doc/examples/CMakeFiles/matrixfree_cg.dir/requires

doc/examples/CMakeFiles/matrixfree_cg.dir/clean:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/matrixfree_cg.dir/cmake_clean.cmake
.PHONY : doc/examples/CMakeFiles/matrixfree_cg.dir/clean

doc/examples/CMakeFiles/matrixfree_cg.dir/depend:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0 /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/doc/examples /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/doc/examples/CMakeFiles/matrixfree_cg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/examples/CMakeFiles/matrixfree_cg.dir/depend

