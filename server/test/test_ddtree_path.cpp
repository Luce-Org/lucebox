#include "common/ddtree.h"
#include "host_check.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using dflash::common::DDTree;
using dflash::common::follow_verified_tree;
using dflash::common::truncate_verified_path;

static int g_checks = 0;

int main() {
    DDTree tree;
    tree.n_nodes = 2;
    tree.token_ids = {11, 22};
    tree.depths = {1, 2};
    tree.parents = {-1, 0, 1};
    tree.child_maps.resize(3);
    tree.child_maps[0][11] = 1;
    tree.child_maps[1][22] = 2;

    const int32_t posterior[] = {11, 22, 33};
    int pending = -1;
    std::vector<int> accepted =
        follow_verified_tree(tree, posterior, pending);
    CHECK((accepted == std::vector<int>{0, 1, 2}));
    CHECK(pending == 33);

    CHECK(truncate_verified_path(accepted, 2, posterior, pending));
    CHECK((accepted == std::vector<int>{0, 1}));
    CHECK(pending == 22);

    CHECK(!truncate_verified_path(accepted, 2, posterior, pending));
    CHECK(pending == 22);

    CHECK(truncate_verified_path(accepted, 0, posterior, pending));
    CHECK(accepted.empty());
    CHECK(pending == -1);

    std::puts("ddtree path tests passed");
    return 0;
}
