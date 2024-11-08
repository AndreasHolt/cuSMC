#include "output_write.h"
using namespace std::chrono;

bool is_goal_node_by_id(const std::unordered_map<int,node*>& node_map, const int id)
{
    return node_map.count(id) && node_map.at(id)->type == node::goal;
}

float output_writer::calc_percentage(const unsigned long long counter, const unsigned long long divisor)
{
    return (static_cast<float>(counter) / static_cast<float>(divisor)) * 100.0f;
}

void output_writer::write_to_file(const result_pointers* results,
    std::chrono::steady_clock::duration sim_duration) const
{
    const unsigned sims_pr_thread = results->sim_per_thread();
    std::ofstream file_node, file_variable;
    file_node.open(this->file_path_ + "_node_data_" + std::to_string(output_counter_) + ".csv" );
    file_variable.open(this->file_path_ + "_variable_data_" + std::to_string(output_counter_) + ".csv");

    file_node << "node_id,name,thread,network_id,reach_count,avg_steps,avg_time\n";
    file_variable << "variableID,name,thread,avg_max_value,max_value\n";

    for (int t = 0; t < results->threads; ++t)
    {
        for (int i = 0; i < this->node_count_; ++i)
        {
            const int index = t * this->node_count_ + i;
            const int id = i + 1;
            const int network_id = this->properties_->node_network->at(id);
            std::string name = this->properties_->node_names->count(id)
                ? this->properties_->node_names->at(id)
                  : "_";

            file_node
                << id << ','
                << name << ','
                << t << ','
                << network_id << ','
                << results->nodes[index].reached << ','
                << results->nodes[index].avg_steps() << ','
                << results->nodes[index].avg_time() << '\n';
        }

        for (int i = 0; i < this->variable_count_; ++i)
        {
            const int index = t * this->variable_count_ + i;
            file_variable
                << i << ','
                << (this->properties_->variable_names->count(i) ? this->properties_->variable_names->at(i) : "_") << ','
                << t << ','
                << results->variables[index].avg_max_value(sims_pr_thread) << ','
                << results->variables[index].max_value << '\n';
        }
    }

    file_node.flush();
    file_variable.flush();

    file_node.close();
    file_variable.close();
}

void insert_name(const int id, const std::unordered_map<int, std::string>* name_map, std::ostream& stream)
{
    stream << abs(id) << ": ";
    if(name_map->count(abs(id)))
    {
        stream << name_map->at(abs(id));
    }
    else
    {
        stream << '_';
    }
}

bool id_sorter(const int lhs, const int rhs)
{
    return lhs < rhs;
}

void output_writer::write_summary_to_stream(std::ostream& stream,
                                            steady_clock::duration sim_duration) const
{
    stream << "\n\nReachability: ";

    // node_summary no_hit = node_summary{0, 0};
    node_summary global_no_hit = {0, 0, 0};
    for(auto& groups : this->network_groups_)
    {
        stream << "\nProcess: " << groups.first << "\n";
        node_summary no_hit = {0, 0, 0};

        for(const int id : groups.second)
        {
            node_summary summary = this->node_summary_map_.at(id);
            if(!is_goal_node_by_id(this->properties_->node_map, id))
                no_hit.cumulative(summary);

            const float percentage = calc_percentage(summary.reach_count, total_simulations_);
            insert_name(id, this->properties_->node_names, stream);
            stream << " reached " << summary.reach_count << " times (" << percentage << "%)\n";
            stream << "    " << " avg steps: " << summary.avg_steps() << " | avg sim time: " << summary.avg_time() << "t.\n";
        }

        const float percentage = calc_percentage(no_hit.reach_count, total_simulations_);
        stream << "\nProcess " << groups.first << " did not reach goal " << no_hit.reach_count << " times ("<< percentage <<"%)\n";
        stream << "    " <<" avg steps: " << no_hit.avg_steps() << " | avg sim time: " << no_hit.avg_time() << "t.\n";
        global_no_hit.cumulative(no_hit);
    }

    stream << "\nAverage maximum value of each variable: \n";

    for (int j = 0; j < this->variable_summaries_.size; ++j)
    {
        const variable_summary& result = this->variable_summaries_.store[j];
        stream << "variable "
               << result.variable_id
               << " (" << (this->properties_->variable_names->count(static_cast<int>(result.variable_id))
                           ? this->properties_->variable_names->at(static_cast<int>(result.variable_id))
                           : "_") << ") "
               <<  " = "
               << result.avg_max_value() << "\n";
    }

    stream << '\n';
    stream << "Probability of false negative results (alpha): " << this->alpha_*100 << "%\n";
    stream << "Probability uncertainty (-+epsilon) of results: " << this->epsilon_*100 << "%\n";
    stream << "Nr of simulations: " << total_simulations_ << "\n\n";
    stream << "Simulation ran for: " << duration_cast<milliseconds>(sim_duration).count()
           << "[ms]" << "\n";
    stream << "Including overhead: " << duration_cast<milliseconds>((steady_clock::now() - properties_->pre_optimisation_start)).count()<< "[ms]\n";
}

void output_writer::write_lite(const steady_clock::duration sim_duration) const
{
    std::ofstream lite;
    const std::string file_path = file_path_ + "_lite_summary.txt";
    lite.open(file_path);

    lite << duration_cast<milliseconds>(sim_duration).count();

    lite.flush();
    lite.close();
}

void output_writer::write_hit_file(const std::chrono::steady_clock::duration sim_duration) const
{
    if (this->node_summary_map_.empty()) return;
    std::ofstream file = std::ofstream(this->file_path_ + "_results.tsv", std::ofstream::out|std::ofstream::trunc);

    bool any = false;
    for (const auto& pair : this->node_summary_map_)
    {
        if (!is_goal_node_by_id(this->properties_->node_map, pair.first)) continue;
        const float percentage = this->calc_percentage(pair.second.reach_count, total_simulations_);
        file << percentage << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(sim_duration).count();
        any = true;
    }
    if(!any)
    {
        file << "0.0\t" << std::chrono::duration_cast<std::chrono::milliseconds>(sim_duration).count();
    }

    file.flush();
    file.close();
}

void output_writer::write_reach_file() const
{
    if(this->node_summary_map_.empty()) return;
    std::ofstream file = std::ofstream(this->file_path_ + "_reach.tsv", std::ofstream::out|std::ofstream::trunc);

    file << "process\treach_perc\n";
    for(auto& groups : this->network_groups_)
    {
        node_summary no_hit = {0, 0, 0};
        for(const int id : groups.second)
        {
            node_summary summary = this->node_summary_map_.at(id);
            if(!is_goal_node_by_id(this->properties_->node_map, id))
                no_hit.cumulative(summary);
        }

        const float percentage = calc_percentage(no_hit.reach_count, total_simulations_);
        file << groups.first << '\t' << percentage << '\n';
    }

    file.flush();
    file.close();
}

void output_writer::setup_network_groups(const sim_config* config)
{
    for(const auto& pair : *config->properties->node_network)
    {
        if(this->network_groups_.count(pair.second))
            this->network_groups_.at(pair.second).push_back(pair.first);
        else
        {
            this->network_groups_.insert(std::pair<int, std::list<int>>(pair.second, std::list<int>()));
            this->network_groups_.at(pair.second).push_back(pair.first);
        }
    }
    for(auto& pair : this->network_groups_)
    {
        pair.second.sort(id_sorter);
    }
}

output_writer::output_writer(const sim_config* config, const network* model)
{
    this->file_path_ = config->paths->output_path;
    this->write_mode_ = config->write_mode;
    this->node_summary_map_ = std::map<int, node_summary>();
    this->total_simulations_ = config->total_simulations() * config->simulation_repetitions;
    this->alpha_ = config->alpha;
    this->epsilon_ = config->epsilon;
    this->properties_ = config->properties;
    this->node_count_ = static_cast<int>(config->node_count);
    this->variable_count_ = static_cast<int>(config->tracked_variable_count);

    this->variable_summaries_ = arr<variable_summary>{
        static_cast<variable_summary*>(malloc(sizeof(variable_summary)*config->tracked_variable_count)),
        static_cast<int>(config->tracked_variable_count)
    };

    for (int i = 0, j = 0; i < model->variables.size; ++i)
    {
        if(!model->variables.store[i].should_track) continue;
        this->variable_summaries_.store[j++] = variable_summary{static_cast<unsigned>(i), 0.0, 0};
    }

    this->network_groups_ = std::map<int, std::list<int>>();
    this->setup_network_groups(config);
}

output_writer::~output_writer()
{
    free(this->variable_summaries_.store);
}

void output_writer::analyse_batch(const result_pointers& pointers)
{
    const unsigned sims_per_thread = pointers.sim_per_thread();
    const int node_count = this->node_count_;

    for (int t = 0; t < pointers.threads; ++t)
    {
        for (int i = 0; i < node_count; ++i)
        {
            const int index = t * node_count + i;
            const int id = i+1; //ids go from 1..*, so to index them, they are id-1. Here we do the reverse.
            const node_results* node = &pointers.nodes[index];
            node_summary summary = node_summary
            {
                node->reached,
                node->total_steps,
                node->total_time
            };

            if(this->node_summary_map_.count(id))
                this->node_summary_map_.at(id).cumulative(summary);
            else
                this->node_summary_map_.insert(
                    std::pair<int, node_summary>(id, summary));
        }

        for (int i = 0; i < this->variable_count_; ++i)
        {
            const int k = t * this->variable_count_ + i;
            variable_summary summary = {
                this->variable_summaries_.store[i].variable_id,
                pointers.variables[k].total_values,
                sims_per_thread
            };
            this->variable_summaries_.store[i].combine(summary);
        }
    }
}

void output_writer::write(const result_store* sim_result, std::chrono::steady_clock::duration sim_duration)
{
    if (this->write_mode_ & lite_sum)
    {
        this->write_lite(sim_duration);
    }

    if(this->write_mode_ & (console_sum | file_sum | file_data | hit_file | reach_file))
    {
        const result_pointers pointers = sim_result->load_results();
        this->analyse_batch(pointers);

        if(this->write_mode_ & file_data) write_to_file(&pointers, sim_duration);
        pointers.free_internals();
    }
    output_counter_++;
    sim_result->clear();
}

void output_writer::write_summary(std::chrono::steady_clock::duration sim_duration) const
{
    if (this->write_mode_ & console_sum)
    {
        this->write_summary_to_stream(std::cout, sim_duration);
    }
    if (this->write_mode_ & file_sum)
    {
        const std::string path = this->file_path_ + "_summary.csv";

        std::ofstream summary;
        summary.open(path);
        this->write_summary_to_stream(summary, sim_duration);

        summary.flush();
        summary.close();
    }
    if (this->write_mode_ & hit_file)
    {
        this->write_hit_file(sim_duration);
    }
    if(this->write_mode_ & reach_file)
    {
        this->write_reach_file();
    }
}

void output_writer::clear()
{
    this->node_summary_map_.clear();
    for (int i = 0; i < this->variable_summaries_.size; ++i)
    {
        this->variable_summaries_.store[i] = variable_summary{
            this->variable_summaries_.store[i].variable_id,
            0.0,
            0,
        };
    }
}