#pragma once

#include "common/sim_config.h"
#include "../simulation/simulation_config.h"
#include "results/result_store.h"
#include  "results/output_write.h"

void convert_config(sim_config* config, const simulation_config* in_config);