cmake_minimum_required(VERSION 3.18)
project(CudaIntro LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## A10 settings
#set(CMAKE_CUDA_ARCHITECTURES 86)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
#    -arch=sm_86 \
#    --maxrregcount=255 \
#    -Xptxas=-v \
#    -Xcompiler=-Wall")
##

# T4 Settings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
##

# Add your source files
add_executable(CudaIntro main.cu
        # Parser
        automata_parser/uppaal_xml_parser.cpp automata_parser/uppaal_xml_parser.h
        automata_parser/declaration_parser.h automata_parser/declaration_parser.cpp
        automata_parser/string_extractor.h automata_parser/string_extractor.cpp
        automata_parser/pugixml.cpp automata_parser
        automata_parser/variable_expression_evaluator.cpp automata_parser/variable_expression_evaluator.h

        # engine
        include/engine/domain.cu include/engine/domain.h

        # Includes
        include/common_macros.h
        include/common_macros.h
        automata_parser/network/network_props.h
        simulation/simulation.cu
        simulation/simulation.cuh
        automata_parser/network/domain_optimization_visitor.h
        automata_parser/network/domain_optimization_visitor.cpp
        automata_parser/network/visitor.cpp
        automata_parser/network/pn_compile_visitor.cpp
        simulation/state/shared_model_state.cu
        simulation/state/shared_model_state.cuh
        main.cu
        main.cuh
        simulation/state/shared_run_state.cu
        simulation/state/shared_run_state.cuh
        automata_parser/variable_usage_visitor.h
        include/variable_types.h
        simulation/statistics.cu
        simulation/statistics.cuh
        automata_parser/network/visitor.h
)


# Enable separable compilation (optional, depends on your needs)

# Add include directories if needed
# target_include_directories(CudaIntro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_include_directories(CudaIntro PRIVATE ${CMAKE_SOURCE_DIR}/UPPAALXMLParser)

# Below doesn't work for now
#find_package(CUDA REQUIRED)
#target_include_directories(CudaIntro PRIVATE ${CUDA_INCLUDE_DIRS})

# A10 Settings
#set_target_properties(CudaIntro PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_ARCHITECTURES 86)
##

# T4 Settings
set_target_properties(CudaIntro PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
##

target_include_directories(CudaIntro PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")

target_compile_options(CudaIntro PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
