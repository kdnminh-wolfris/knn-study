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
include test/CMakeFiles/stdvector_4.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/stdvector_4.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/stdvector_4.dir/flags.make

test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o: test/CMakeFiles/stdvector_4.dir/flags.make
test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o: /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/stdvector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stdvector_4.dir/stdvector.cpp.o -c /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/stdvector.cpp

test/CMakeFiles/stdvector_4.dir/stdvector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stdvector_4.dir/stdvector.cpp.i"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/stdvector.cpp > CMakeFiles/stdvector_4.dir/stdvector.cpp.i

test/CMakeFiles/stdvector_4.dir/stdvector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stdvector_4.dir/stdvector.cpp.s"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test/stdvector.cpp -o CMakeFiles/stdvector_4.dir/stdvector.cpp.s

test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o.requires:

.PHONY : test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o.requires

test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o.provides: test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/stdvector_4.dir/build.make test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o.provides.build
.PHONY : test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o.provides

test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o.provides.build: test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o


# Object files for target stdvector_4
stdvector_4_OBJECTS = \
"CMakeFiles/stdvector_4.dir/stdvector.cpp.o"

# External object files for target stdvector_4
stdvector_4_EXTERNAL_OBJECTS =

test/stdvector_4: test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o
test/stdvector_4: test/CMakeFiles/stdvector_4.dir/build.make
test/stdvector_4: test/CMakeFiles/stdvector_4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable stdvector_4"
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stdvector_4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/stdvector_4.dir/build: test/stdvector_4

.PHONY : test/CMakeFiles/stdvector_4.dir/build

test/CMakeFiles/stdvector_4.dir/requires: test/CMakeFiles/stdvector_4.dir/stdvector.cpp.o.requires

.PHONY : test/CMakeFiles/stdvector_4.dir/requires

test/CMakeFiles/stdvector_4.dir/clean:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test && $(CMAKE_COMMAND) -P CMakeFiles/stdvector_4.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/stdvector_4.dir/clean

test/CMakeFiles/stdvector_4.dir/depend:
	cd /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0 /mnt/d/Projects/knn-study/eigen-3.4.0/eigen-3.4.0/test /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test /mnt/d/Projects/knn-study/eigen-3.4.0/build_dir/test/CMakeFiles/stdvector_4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/stdvector_4.dir/depend

