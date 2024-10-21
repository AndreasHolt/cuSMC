cmake_minimum_required(VERSION 3.18)  # Minimum version for CUDA support
project(CudaIntro LANGUAGES CUDA CXX)   # Specify CUDA and C++ as the primary languages

set(CMAKE_CUDA_STANDARD 17)              # Set the CUDA standard to C++17
set(CMAKE_CXX_STANDARD 17)               # Set the C++ standard to C++17
set(CMAKE_CUDA_STANDARD_REQUIRED ON)     # Ensure that the specified standard is required
set(CMAKE_CXX_STANDARD_REQUIRED ON)      # Ensure that the specified standard is required

# Add your source files
add_executable(CudaIntro parse.cpp
        # Parser
        automata_parser/uppaal_xml_parser.cpp automata_parser/uppaal_xml_parser.h
        automata_parser/declaration_parser.h automata_parser/declaration_parser.cpp
        automata_parser/string_extractor.h automata_parser/string_extractor.cpp
        automata_parser/pugixml.cpp automata_parser automata_parser/smacc_parser.cpp
        automata_parser/variable_expression_evaluator.cpp automata_parser/variable_expression_evaluator.h

        # Engine
        engine/Domain.cu engine/Domain.h
        engine/model_oracle.cu engine/model_oracle.h
        engine/automata_engine.cu


        # Includes
        include/common_macros.h
        include/common_macros.h
        engine/interpreter.h)


# Enable separable compilation (optional, depends on your needs)
set_target_properties(CudaIntro PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add include directories if needed
# target_include_directories(CudaIntro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_include_directories(CudaIntro PRIVATE ${CMAKE_SOURCE_DIR}/UPPAALXMLParser)

# Below doesn't work for now
#find_package(CUDA REQUIRED)
#target_include_directories(CudaIntro PRIVATE ${CUDA_INCLUDE_DIRS})

target_include_directories(CudaIntro PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")


# Link libraries if needed
# target_link_libraries(CudaIntro PRIVATE some_library)

# Set compile options for CUDA and C++
target_compile_options(CudaIntro PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        #    $<$<COMPILE_LANGUAGE:CXX>:-Wall -pedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)
