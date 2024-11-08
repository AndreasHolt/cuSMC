#pragma once
#include <array>
#include <string>
#include <map>
#include "result_store.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include "../common/sim_config.h"

enum write_modes //values must be powers of 2.
{
    console_sum = 1,
    file_sum = 2,
    file_data = 4,
    lite_sum = 8,
    model_out = 32,
    pretty_out = 64,
    hit_file = 128,
    reach_file = 256,
};

struct node_summary
{
    unsigned int reach_count = 0;
    unsigned long total_steps = 0;
    double total_time = 0.0;
    // double avg_steps = 0;
    // double avg_time = 0;

    double avg_steps() const
    {
        if(reach_count == 0) return 0.0;
        return static_cast<double>(total_steps) / static_cast<double>(reach_count);
    }

    double avg_time() const
    {
        if(reach_count == 0) return 0.0;
        return total_time / static_cast<double>(reach_count);
    }

    void cumulative(const node_summary& other)
    {
        this->total_steps += other.total_steps;
        this->total_time += other.total_time;
        this->reach_count += other.reach_count;
    }
};

struct variable_summary
{
    unsigned int variable_id;
    double sum_max_values;
    unsigned long values_counted;

    void combine(const variable_summary& other)
    {
        if(this->variable_id != other.variable_id)
            throw std::runtime_error("VARIABLE ID mismatch when combining summaries");
        this->sum_max_values += other.sum_max_values;
        this->values_counted += other.values_counted;
    }

    double avg_max_value() const
    {
        return sum_max_values / static_cast<double>(values_counted);
    }
};


struct output_properties
{
    std::chrono::steady_clock::time_point pre_optimisation_start;
    std::chrono::steady_clock::time_point post_optimisation_start;
    std::unordered_map<int, std::string>* variable_names;
    std::unordered_map<int, std::string>* node_names;
    std::unordered_map<int, std::string>* template_names;
    std::unordered_map<int, int>* node_network;
    std::unordered_map<int, node*> node_map;
};


class output_writer
{
private:
    std::string file_path_;
    int write_mode_;

    int node_count_;
    int variable_count_;
    size_t total_simulations_;
    unsigned output_counter_ = 0;
    std::map<int, node_summary> node_summary_map_{};
    std::map<int, std::list<int>> network_groups_;
    arr<variable_summary> variable_summaries_ = arr<variable_summary>::empty();
    double epsilon_;
    double alpha_;
    output_properties* properties_;

    static float calc_percentage(const unsigned long long counter, const unsigned long long divisor);

    void write_to_file(const result_pointers* results,
        std::chrono::steady_clock::duration sim_duration) const;

    void write_summary_to_stream(std::ostream& stream, std::chrono::steady_clock::duration sim_duration) const;
    void write_lite(std::chrono::steady_clock::duration sim_duration) const;
    void write_hit_file(std::chrono::steady_clock::duration sim_duration) const;
    void write_reach_file() const;
    void setup_network_groups(const sim_config* config);

public:
    explicit output_writer(const sim_config* config, const network* model);
    ~output_writer();
    void analyse_batch(const result_pointers& pointers);
    void write(const result_store* sim_result, std::chrono::steady_clock::duration sim_duration);
    void write_summary(std::chrono::steady_clock::duration sim_duration) const;
    void clear();
    static int parse_mode(const std::string& str);
};



inline int output_writer::parse_mode(const std::string& str)
{
    if(str.empty()) return 0;

    int mode = 0;
    if(str.find('l') != std::string::npos) mode |= lite_sum;
    if(str.find('c') != std::string::npos) mode |= console_sum;
    if(str.find('f') != std::string::npos) mode |= file_sum;
    if(str.find('d') != std::string::npos) mode |= file_data;
    if(str.find('m') != std::string::npos) mode |= model_out;
    if(str.find('p') != std::string::npos) mode |= pretty_out;
    if(str.find('r') != std::string::npos) mode |= hit_file;
    if(str.find('q') != std::string::npos) mode |= reach_file;

    return mode;
}