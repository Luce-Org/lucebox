// cluster_worker_main.h - lockstep worker loop for ranks 1..N-1
// (maikzz32/lucebox-halo-cluster). server_main builds the backend through the
// regular factory (the DeepSeek4 backend loads only this rank's expert share)
// and hands it over; the worker joins the head, brings up RCCL and replays
// every request the head broadcasts. Returns the process exit code.

#pragma once

#include "cluster/cluster_config.h"
#include "common/model_backend.h"

#include <memory>
#include <string>

namespace luce::cluster {

int run_cluster_worker(const ClusterConfig & cfg,
                       std::unique_ptr<common::ModelBackend> backend,
                       const std::string & model_path,
                       int device);

}  // namespace luce::cluster
