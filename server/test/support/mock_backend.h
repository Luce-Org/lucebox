#pragma once
#include "common/model_backend.h"

namespace {
struct MockBackend : dflash::common::ModelBackend {
    void print_ready_banner() const override {}
    bool park(dflash::common::ParkTarget) override { return true; }
    bool unpark(dflash::common::ParkTarget) override { return true; }
    bool is_target_parked() const override { return false; }
    dflash::common::GenerateResult generate_impl(const dflash::common::GenerateRequest &, const dflash::common::DaemonIO &) override { return {}; }
    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int  snapshot_cur_pos(int) const override { return 0; }
    dflash::common::GenerateResult restore_and_generate_impl(int, const dflash::common::GenerateRequest &, const dflash::common::DaemonIO &) override { return {}; }
    bool handle_compress(const std::string &, const dflash::common::DaemonIO &) override { return false; }
    void free_drafter() override {}
    void shutdown() override {}
};

}
