//
// Created by andwh on 13/12/2024.
//

#ifndef CSV_WRITER_CUH
#define CSV_WRITER_CUH
#include <string>


void writeTimingToCSV(const std::string& xml_path, int MC, int simulations, double timeBound, double timing_ms);



#endif //CSV_WRITER_CUH
