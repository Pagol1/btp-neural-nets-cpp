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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/neural_network.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/neural_network.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/neural_network.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neural_network.dir/flags.make

CMakeFiles/neural_network.dir/src/MNIST.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/src/MNIST.cpp.o: /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/MNIST.cpp
CMakeFiles/neural_network.dir/src/MNIST.cpp.o: CMakeFiles/neural_network.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neural_network.dir/src/MNIST.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network.dir/src/MNIST.cpp.o -MF CMakeFiles/neural_network.dir/src/MNIST.cpp.o.d -o CMakeFiles/neural_network.dir/src/MNIST.cpp.o -c /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/MNIST.cpp

CMakeFiles/neural_network.dir/src/MNIST.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/src/MNIST.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/MNIST.cpp > CMakeFiles/neural_network.dir/src/MNIST.cpp.i

CMakeFiles/neural_network.dir/src/MNIST.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/src/MNIST.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/MNIST.cpp -o CMakeFiles/neural_network.dir/src/MNIST.cpp.s

CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o: /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/activation_functions/Sigmoid.cpp
CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o: CMakeFiles/neural_network.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o -MF CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o.d -o CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o -c /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/activation_functions/Sigmoid.cpp

CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/activation_functions/Sigmoid.cpp > CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.i

CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/activation_functions/Sigmoid.cpp -o CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.s

CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o: /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/activation_functions/Softmax.cpp
CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o: CMakeFiles/neural_network.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o -MF CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o.d -o CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o -c /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/activation_functions/Softmax.cpp

CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/activation_functions/Softmax.cpp > CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.i

CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/activation_functions/Softmax.cpp -o CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.s

CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o: /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/layers/DenseLayer.cpp
CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o: CMakeFiles/neural_network.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o -MF CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o.d -o CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o -c /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/layers/DenseLayer.cpp

CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/layers/DenseLayer.cpp > CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.i

CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/layers/DenseLayer.cpp -o CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.s

CMakeFiles/neural_network.dir/src/main.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/src/main.cpp.o: /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/main.cpp
CMakeFiles/neural_network.dir/src/main.cpp.o: CMakeFiles/neural_network.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/neural_network.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network.dir/src/main.cpp.o -MF CMakeFiles/neural_network.dir/src/main.cpp.o.d -o CMakeFiles/neural_network.dir/src/main.cpp.o -c /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/main.cpp

CMakeFiles/neural_network.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/main.cpp > CMakeFiles/neural_network.dir/src/main.cpp.i

CMakeFiles/neural_network.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/main.cpp -o CMakeFiles/neural_network.dir/src/main.cpp.s

CMakeFiles/neural_network.dir/src/models/ANN.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/src/models/ANN.cpp.o: /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/models/ANN.cpp
CMakeFiles/neural_network.dir/src/models/ANN.cpp.o: CMakeFiles/neural_network.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/neural_network.dir/src/models/ANN.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/neural_network.dir/src/models/ANN.cpp.o -MF CMakeFiles/neural_network.dir/src/models/ANN.cpp.o.d -o CMakeFiles/neural_network.dir/src/models/ANN.cpp.o -c /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/models/ANN.cpp

CMakeFiles/neural_network.dir/src/models/ANN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/src/models/ANN.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/models/ANN.cpp > CMakeFiles/neural_network.dir/src/models/ANN.cpp.i

CMakeFiles/neural_network.dir/src/models/ANN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/src/models/ANN.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/src/models/ANN.cpp -o CMakeFiles/neural_network.dir/src/models/ANN.cpp.s

# Object files for target neural_network
neural_network_OBJECTS = \
"CMakeFiles/neural_network.dir/src/MNIST.cpp.o" \
"CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o" \
"CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o" \
"CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o" \
"CMakeFiles/neural_network.dir/src/main.cpp.o" \
"CMakeFiles/neural_network.dir/src/models/ANN.cpp.o"

# External object files for target neural_network
neural_network_EXTERNAL_OBJECTS =

neural_network: CMakeFiles/neural_network.dir/src/MNIST.cpp.o
neural_network: CMakeFiles/neural_network.dir/src/activation_functions/Sigmoid.cpp.o
neural_network: CMakeFiles/neural_network.dir/src/activation_functions/Softmax.cpp.o
neural_network: CMakeFiles/neural_network.dir/src/layers/DenseLayer.cpp.o
neural_network: CMakeFiles/neural_network.dir/src/main.cpp.o
neural_network: CMakeFiles/neural_network.dir/src/models/ANN.cpp.o
neural_network: CMakeFiles/neural_network.dir/build.make
neural_network: CMakeFiles/neural_network.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable neural_network"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neural_network.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neural_network.dir/build: neural_network
.PHONY : CMakeFiles/neural_network.dir/build

CMakeFiles/neural_network.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neural_network.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neural_network.dir/clean

CMakeFiles/neural_network.dir/depend:
	cd /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build /home/pagol1/Stuff/College/BTP/ANNs/neural_network_cpp/build/CMakeFiles/neural_network.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/neural_network.dir/depend

