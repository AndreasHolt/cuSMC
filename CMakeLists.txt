cmake_minimum_required(VERSION 3.18)
project(cuSMC LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

add_executable(cuSMC main.cu
        # Program entrypoint
        main.cu
        main.cuh

        #Parser
        automata_parser/uppaal_xml_parser.cpp automata_parser/uppaal_xml_parser.h
        automata_parser/declaration_parser.h automata_parser/declaration_parser.cpp
        automata_parser/string_extractor.h automata_parser/string_extractor.cpp
        automata_parser/pugixml.cpp automata_parser
        automata_parser/variable_expression_evaluator.cpp automata_parser/variable_expression_evaluator.h
        automata_parser/variable_usage_visitor.h
        automata_parser/network/domain_optimization_visitor.h
        automata_parser/network/domain_optimization_visitor.cpp
        automata_parser/network/visitor.cpp
        automata_parser/network/pn_compile_visitor.cpp
        automata_parser/network/network_props.h
        automata_parser/network/visitor.h

        # Includes
        include/engine/domain.cu include/engine/domain.h
        include/common_macros.h
        include/common_macros.h
        include/variable_types.h

        # Simulation
        simulation/simulation.cu
        simulation/simulation.cuh
        simulation/expressions.cu
        simulation/expressions.cuh

        # Statistics/logging
        simulation/statistics.cu
        simulation/statistics.cuh

        # Data Structures
        simulation/state/shared_model_state.cu
        simulation/state/shared_model_state.cuh
        simulation/state/shared_run_state.cu
        simulation/state/shared_run_state.cuh
        simulation/state/model_verification.cu
        simulation/state/model_verification.cuh
        smc.cu
        smc.cuh
)

set_target_properties(cuSMC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuSMC PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")

target_compile_options(cuSMC PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
