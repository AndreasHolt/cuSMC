//
// Created by andwh on 13/12/2024.
//

#ifndef CSV_WRITER_CUH
#define CSV_WRITER_CUH
#include <string>
#include "statistics.cuh"


void writeTimingToCSV(const std::string& xml_path, int MC, int simulations, double timeBound, double timing_ms);
void write_var_at_time_array_to_csv(const var_at_time* data, int simulations, int *size, const std::string& filename);


#endif //CSV_WRITER_CUH
