//
// Created by andwh on 13/12/2024.
//

#include "write_csv.cuh"

#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <filesystem>
#include <iostream>
#include <set>

#include "statistics.cuh"

void write_var_at_time_array_to_csv(const var_at_time* data, int simulations, int *size, const std::string& filename) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path project_root = current_path.parent_path();

    for (int sims = 0; sims < simulations; sims++) {
        const std::string folder = (project_root /"Graph"/ std::to_string(sims)).string();
        std::filesystem::create_directories(folder);
    }
    for (int sims = 0; sims < simulations; sims++) {
        const std::string path = (project_root /"Graph" / std::to_string(sims) / filename).string();
        std::ofstream csv_file(path);
        if (!csv_file.is_open()) {
            std::cerr << "Error: Could not open file at path: " << path << std::endl;
            return;
        }

        // Write the header row
        csv_file << "Value,Time\n";

        // Write each data point as a new row
        for (int i = 0; i < size[sims]; i++) {
            int index = sims * VAR_OVER_TIME_ARRAY_SIZE + i;
            csv_file << data[index].value << "," << data[index].time << "\n";
        }

        csv_file.close();
    }

}

void writeTimingToCSV(const std::string& xml_path, int MC, int simulations, double timeBound, double timing_ms) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path project_root = current_path.parent_path();
    const std::string CSV_FILE = (project_root / "results5.csv").string();

    std::filesystem::path p(xml_path);
    std::string filename = p.filename().string();

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << timeBound;
    std::string config = "MC" + std::to_string(MC) +
                        "_Sim" + std::to_string(simulations) +
                        "_Time" + ss.str();

    // Read existing CSV content
    std::vector<std::vector<std::string>> csv_data;
    if (std::filesystem::exists(CSV_FILE)) {
        std::ifstream file(CSV_FILE);
        std::string line;
        while (std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ';')) {
                row.push_back(cell);
            }
            csv_data.push_back(row);
        }
    }

    // Initialize empty CSV if needed
    if (csv_data.empty()) {
        csv_data.push_back(std::vector<std::string>{""});
    }

    // Find or add config in first row (starting from column B)
    int config_col = -1;
    for (size_t col = 1; col < csv_data[0].size(); col++) {
        if (csv_data[0][col] == config) {
            config_col = col;
            break;
        }
    }
    if (config_col == -1) {
        config_col = csv_data[0].size();
        csv_data[0].push_back(config);
    }

    // Find or add filename row
    int file_row = -1;
    for (size_t row = 1; row < csv_data.size(); row++) {
        if (!csv_data[row].empty() && csv_data[row][0] == filename) {
            file_row = row;
            break;
        }
    }
    if (file_row == -1) {
        file_row = csv_data.size();
        csv_data.push_back(std::vector<std::string>{filename});
    }

    for (auto& row : csv_data) {
        while (row.size() <= config_col) {
            row.push_back("");
        }
    }

    // Write the timing value
    csv_data[file_row][config_col] = std::to_string(timing_ms);

    // Write to CSV file
    std::ofstream outfile(CSV_FILE);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing" << std::endl;
        return;
    }

    for (const auto& row : csv_data) {
        outfile << row[0];
        for (size_t i = 1; i < row.size(); i++) {
            outfile << ";" << row[i];
        }
        outfile << "\n";
    }
}



