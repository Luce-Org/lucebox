#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_placement.h"
#include "../src/common/platform_env.h"

#include <cstdlib>
#include <string>

using namespace dflash::common;

namespace {

struct MoeExpertOwnerPlacementFixture {};

class ScopedEnvironment {
public:
    explicit ScopedEnvironment(const char * name) : name_(name) {
        const char * value = std::getenv(name);
        if (value) {
            existed_ = true;
            value_ = value;
        }
    }

    ~ScopedEnvironment() {
        if (existed_) {
            set_environment_variable(name_.c_str(), value_.c_str(), true);
        } else {
            unset_environment_variable(name_.c_str());
        }
    }

private:
    std::string name_;
    std::string value_;
    bool existed_ = false;
};

}  // namespace

TEST_CASE(MoeExpertOwnerPlacementFixture, resolves_programmatic_and_environment_owners) {
    ScopedEnvironment generic_gpu("DFLASH_MOE_TP_GPU");
    ScopedEnvironment ds4_gpu("DFLASH_DS4_MOE_TP_GPU");
    ScopedEnvironment ipc_gpu("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU");
    REQUIRE(unset_environment_variable("DFLASH_MOE_TP_GPU") == 0);
    REQUIRE(unset_environment_variable("DFLASH_DS4_MOE_TP_GPU") == 0);
    REQUIRE(unset_environment_variable(
                "DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU") == 0);

    MoeExpertOwnerPlacement owner;
    std::string error;
    REQUIRE(resolve_moe_expert_owner_placement(0, -1, owner, &error));
    REQUIRE(owner.primary_gpu == 0);
    REQUIRE(owner.expert_gpu == 0);
    REQUIRE(!owner.heterogeneous());

    REQUIRE(resolve_moe_expert_owner_placement(0, 1, owner, &error));
    REQUIRE(owner.expert_gpu == 1);
    REQUIRE(owner.heterogeneous());
    REQUIRE(!resolve_moe_expert_owner_placement(-1, 0, owner, &error));
    REQUIRE(!resolve_moe_expert_owner_placement(0, -2, owner, &error));

    REQUIRE(set_environment_variable("DFLASH_MOE_TP_GPU", "2", true) == 0);
    REQUIRE(set_environment_variable("DFLASH_DS4_MOE_TP_GPU", "3", true) == 0);
    REQUIRE(set_environment_variable(
                "DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU", "4", true) == 0);
    REQUIRE(resolve_moe_expert_owner_placement(0, -1, owner, &error));
    REQUIRE(owner.expert_gpu == 2);

    REQUIRE(unset_environment_variable("DFLASH_MOE_TP_GPU") == 0);
    REQUIRE(resolve_moe_expert_owner_placement(0, -1, owner, &error));
    REQUIRE(owner.expert_gpu == 3);
    REQUIRE(unset_environment_variable("DFLASH_DS4_MOE_TP_GPU") == 0);
    REQUIRE(resolve_moe_expert_owner_placement(0, -1, owner, &error));
    REQUIRE(owner.expert_gpu == 4);

    REQUIRE(set_environment_variable(
                "DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU",
                "not-a-device", true) == 0);
    REQUIRE(!resolve_moe_expert_owner_placement(0, -1, owner, &error));

    // An explicit API choice must not be overridden by process environment.
    REQUIRE(resolve_moe_expert_owner_placement(0, 1, owner, &error));
    REQUIRE(owner.expert_gpu == 1);
}
