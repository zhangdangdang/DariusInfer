# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /usr/local/share/cmake-3.27.1/bin/cmake

# The command to remove a file.
RM = /usr/local/share/cmake-3.27.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/work/interfer/DariusInfer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/work/interfer/DariusInfer/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/darius.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/darius.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/darius.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/darius.dir/flags.make

CMakeFiles/darius.dir/main.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/main.cpp.o: /home/work/interfer/DariusInfer/main.cpp
CMakeFiles/darius.dir/main.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/darius.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/main.cpp.o -MF CMakeFiles/darius.dir/main.cpp.o.d -o CMakeFiles/darius.dir/main.cpp.o -c /home/work/interfer/DariusInfer/main.cpp

CMakeFiles/darius.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/main.cpp > CMakeFiles/darius.dir/main.cpp.i

CMakeFiles/darius.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/main.cpp -o CMakeFiles/darius.dir/main.cpp.s

CMakeFiles/darius.dir/test/axby.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/test/axby.cpp.o: /home/work/interfer/DariusInfer/test/axby.cpp
CMakeFiles/darius.dir/test/axby.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/darius.dir/test/axby.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/test/axby.cpp.o -MF CMakeFiles/darius.dir/test/axby.cpp.o.d -o CMakeFiles/darius.dir/test/axby.cpp.o -c /home/work/interfer/DariusInfer/test/axby.cpp

CMakeFiles/darius.dir/test/axby.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/test/axby.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/test/axby.cpp > CMakeFiles/darius.dir/test/axby.cpp.i

CMakeFiles/darius.dir/test/axby.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/test/axby.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/test/axby.cpp -o CMakeFiles/darius.dir/test/axby.cpp.s

CMakeFiles/darius.dir/test/test1.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/test/test1.cpp.o: /home/work/interfer/DariusInfer/test/test1.cpp
CMakeFiles/darius.dir/test/test1.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/darius.dir/test/test1.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/test/test1.cpp.o -MF CMakeFiles/darius.dir/test/test1.cpp.o.d -o CMakeFiles/darius.dir/test/test1.cpp.o -c /home/work/interfer/DariusInfer/test/test1.cpp

CMakeFiles/darius.dir/test/test1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/test/test1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/test/test1.cpp > CMakeFiles/darius.dir/test/test1.cpp.i

CMakeFiles/darius.dir/test/test1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/test/test1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/test/test1.cpp -o CMakeFiles/darius.dir/test/test1.cpp.s

CMakeFiles/darius.dir/test/test_create_tensor.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/test/test_create_tensor.cpp.o: /home/work/interfer/DariusInfer/test/test_create_tensor.cpp
CMakeFiles/darius.dir/test/test_create_tensor.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/darius.dir/test/test_create_tensor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/test/test_create_tensor.cpp.o -MF CMakeFiles/darius.dir/test/test_create_tensor.cpp.o.d -o CMakeFiles/darius.dir/test/test_create_tensor.cpp.o -c /home/work/interfer/DariusInfer/test/test_create_tensor.cpp

CMakeFiles/darius.dir/test/test_create_tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/test/test_create_tensor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/test/test_create_tensor.cpp > CMakeFiles/darius.dir/test/test_create_tensor.cpp.i

CMakeFiles/darius.dir/test/test_create_tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/test/test_create_tensor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/test/test_create_tensor.cpp -o CMakeFiles/darius.dir/test/test_create_tensor.cpp.s

CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o: /home/work/interfer/DariusInfer/test/test_fill_reshape.cpp
CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o -MF CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o.d -o CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o -c /home/work/interfer/DariusInfer/test/test_fill_reshape.cpp

CMakeFiles/darius.dir/test/test_fill_reshape.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/test/test_fill_reshape.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/test/test_fill_reshape.cpp > CMakeFiles/darius.dir/test/test_fill_reshape.cpp.i

CMakeFiles/darius.dir/test/test_fill_reshape.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/test/test_fill_reshape.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/test/test_fill_reshape.cpp -o CMakeFiles/darius.dir/test/test_fill_reshape.cpp.s

CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o: /home/work/interfer/DariusInfer/test/test_flatten_padding.cpp
CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o -MF CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o.d -o CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o -c /home/work/interfer/DariusInfer/test/test_flatten_padding.cpp

CMakeFiles/darius.dir/test/test_flatten_padding.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/test/test_flatten_padding.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/test/test_flatten_padding.cpp > CMakeFiles/darius.dir/test/test_flatten_padding.cpp.i

CMakeFiles/darius.dir/test/test_flatten_padding.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/test/test_flatten_padding.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/test/test_flatten_padding.cpp -o CMakeFiles/darius.dir/test/test_flatten_padding.cpp.s

CMakeFiles/darius.dir/test/test_get_size.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/test/test_get_size.cpp.o: /home/work/interfer/DariusInfer/test/test_get_size.cpp
CMakeFiles/darius.dir/test/test_get_size.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/darius.dir/test/test_get_size.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/test/test_get_size.cpp.o -MF CMakeFiles/darius.dir/test/test_get_size.cpp.o.d -o CMakeFiles/darius.dir/test/test_get_size.cpp.o -c /home/work/interfer/DariusInfer/test/test_get_size.cpp

CMakeFiles/darius.dir/test/test_get_size.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/test/test_get_size.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/test/test_get_size.cpp > CMakeFiles/darius.dir/test/test_get_size.cpp.i

CMakeFiles/darius.dir/test/test_get_size.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/test/test_get_size.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/test/test_get_size.cpp -o CMakeFiles/darius.dir/test/test_get_size.cpp.s

CMakeFiles/darius.dir/test/test_get_values.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/test/test_get_values.cpp.o: /home/work/interfer/DariusInfer/test/test_get_values.cpp
CMakeFiles/darius.dir/test/test_get_values.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/darius.dir/test/test_get_values.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/test/test_get_values.cpp.o -MF CMakeFiles/darius.dir/test/test_get_values.cpp.o.d -o CMakeFiles/darius.dir/test/test_get_values.cpp.o -c /home/work/interfer/DariusInfer/test/test_get_values.cpp

CMakeFiles/darius.dir/test/test_get_values.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/test/test_get_values.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/test/test_get_values.cpp > CMakeFiles/darius.dir/test/test_get_values.cpp.i

CMakeFiles/darius.dir/test/test_get_values.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/test/test_get_values.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/test/test_get_values.cpp -o CMakeFiles/darius.dir/test/test_get_values.cpp.s

CMakeFiles/darius.dir/test/test_transform.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/test/test_transform.cpp.o: /home/work/interfer/DariusInfer/test/test_transform.cpp
CMakeFiles/darius.dir/test/test_transform.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/darius.dir/test/test_transform.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/test/test_transform.cpp.o -MF CMakeFiles/darius.dir/test/test_transform.cpp.o.d -o CMakeFiles/darius.dir/test/test_transform.cpp.o -c /home/work/interfer/DariusInfer/test/test_transform.cpp

CMakeFiles/darius.dir/test/test_transform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/test/test_transform.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/test/test_transform.cpp > CMakeFiles/darius.dir/test/test_transform.cpp.i

CMakeFiles/darius.dir/test/test_transform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/test/test_transform.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/test/test_transform.cpp -o CMakeFiles/darius.dir/test/test_transform.cpp.s

CMakeFiles/darius.dir/source/tensor.cpp.o: CMakeFiles/darius.dir/flags.make
CMakeFiles/darius.dir/source/tensor.cpp.o: /home/work/interfer/DariusInfer/source/tensor.cpp
CMakeFiles/darius.dir/source/tensor.cpp.o: CMakeFiles/darius.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/darius.dir/source/tensor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/darius.dir/source/tensor.cpp.o -MF CMakeFiles/darius.dir/source/tensor.cpp.o.d -o CMakeFiles/darius.dir/source/tensor.cpp.o -c /home/work/interfer/DariusInfer/source/tensor.cpp

CMakeFiles/darius.dir/source/tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/darius.dir/source/tensor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/work/interfer/DariusInfer/source/tensor.cpp > CMakeFiles/darius.dir/source/tensor.cpp.i

CMakeFiles/darius.dir/source/tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/darius.dir/source/tensor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/work/interfer/DariusInfer/source/tensor.cpp -o CMakeFiles/darius.dir/source/tensor.cpp.s

# Object files for target darius
darius_OBJECTS = \
"CMakeFiles/darius.dir/main.cpp.o" \
"CMakeFiles/darius.dir/test/axby.cpp.o" \
"CMakeFiles/darius.dir/test/test1.cpp.o" \
"CMakeFiles/darius.dir/test/test_create_tensor.cpp.o" \
"CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o" \
"CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o" \
"CMakeFiles/darius.dir/test/test_get_size.cpp.o" \
"CMakeFiles/darius.dir/test/test_get_values.cpp.o" \
"CMakeFiles/darius.dir/test/test_transform.cpp.o" \
"CMakeFiles/darius.dir/source/tensor.cpp.o"

# External object files for target darius
darius_EXTERNAL_OBJECTS =

darius: CMakeFiles/darius.dir/main.cpp.o
darius: CMakeFiles/darius.dir/test/axby.cpp.o
darius: CMakeFiles/darius.dir/test/test1.cpp.o
darius: CMakeFiles/darius.dir/test/test_create_tensor.cpp.o
darius: CMakeFiles/darius.dir/test/test_fill_reshape.cpp.o
darius: CMakeFiles/darius.dir/test/test_flatten_padding.cpp.o
darius: CMakeFiles/darius.dir/test/test_get_size.cpp.o
darius: CMakeFiles/darius.dir/test/test_get_values.cpp.o
darius: CMakeFiles/darius.dir/test/test_transform.cpp.o
darius: CMakeFiles/darius.dir/source/tensor.cpp.o
darius: CMakeFiles/darius.dir/build.make
darius: /usr/local/lib/libglog.so.0.7.0
darius: /usr/local/lib/libgtest.a
darius: /usr/lib/x86_64-linux-gnu/libarmadillo.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_core.so
darius: /usr/lib/x86_64-linux-gnu/libiomp5.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_core.so
darius: /usr/lib/x86_64-linux-gnu/libiomp5.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_core.so
darius: /usr/lib/x86_64-linux-gnu/libiomp5.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so
darius: /usr/lib/x86_64-linux-gnu/libmkl_core.so
darius: /usr/lib/x86_64-linux-gnu/libiomp5.so
darius: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
darius: /usr/lib/x86_64-linux-gnu/libpthread.so
darius: CMakeFiles/darius.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable darius"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/darius.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/darius.dir/build: darius
.PHONY : CMakeFiles/darius.dir/build

CMakeFiles/darius.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/darius.dir/cmake_clean.cmake
.PHONY : CMakeFiles/darius.dir/clean

CMakeFiles/darius.dir/depend:
	cd /home/work/interfer/DariusInfer/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/work/interfer/DariusInfer /home/work/interfer/DariusInfer /home/work/interfer/DariusInfer/cmake-build-debug /home/work/interfer/DariusInfer/cmake-build-debug /home/work/interfer/DariusInfer/cmake-build-debug/CMakeFiles/darius.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/darius.dir/depend

