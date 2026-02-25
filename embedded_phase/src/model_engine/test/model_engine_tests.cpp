#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "model_engine.hpp"

namespace {

void test_node_resource_and_node_layout() {
    eml::If_node_resource resource;
    const bool ok = resource.set_node_layouts(3, 6, 16, 15, 5);
    assert(ok);
    assert(resource.bits_per_node() <= 64);

    eml::IsoNode node;
    node.set_split(resource, 17u, 5u, 321u);
    assert(!node.is_leaf());
    assert(node.feature_id(resource) == 17u);
    assert(node.threshold_slot(resource) == 5u);
    assert(node.left_child(resource) == 321u);

    node.set_leaf(resource, 23u, 7u);
    assert(node.is_leaf());
    assert(node.leaf_size(resource) == 23u);
    assert(node.leaf_depth(resource) == 7u);
}

void test_quantized_iforest_scoring_order() {
    std::vector<uint8_t> train_matrix = {
        0, 0, 1, 1,
        0, 1, 1, 0,
        1, 0, 0, 1,
        1, 1, 0, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 0, 0, 1,
        1, 1, 1, 0,
    };

    eml::If_config cfg;
    cfg.isLoaded = true;
    cfg.num_features = 4;
    cfg.quantization_bits = 2;
    cfg.threshold_bits = 2;
    cfg.feature_bits = 2;
    cfg.child_bits = 8;
    cfg.leaf_size_bits = 4;
    cfg.depth_bits = 4;
    cfg.max_depth = 8;
    cfg.max_nodes_per_tree = 255;
    cfg.max_samples = 1.0f;
    cfg.max_samples_per_tree = 8;
    cfg.n_estimators = 32;
    cfg.bootstrap = false;
    cfg.random_state = 42;
    cfg.threshold_offset = -0.5f;

    eml::IsoForest forest;
    const bool init_ok = forest.init_from_config(cfg);
    assert(init_ok);

    const std::filesystem::path tmp_nml = std::filesystem::temp_directory_path() / "if_test_model_engine.nml";
    {
        std::ofstream fout(tmp_nml, std::ios::binary | std::ios::trunc);
        assert(fout.is_open());

        const uint32_t num_samples = 8u;
        const uint16_t num_features = 4u;
        fout.write(reinterpret_cast<const char*>(&num_samples), sizeof(num_samples));
        fout.write(reinterpret_cast<const char*>(&num_features), sizeof(num_features));

        for (size_t row = 0; row < num_samples; ++row) {
            const uint8_t label = 0u;
            uint8_t packed_features = 0u;
            for (uint16_t col = 0; col < num_features; ++col) {
                const uint8_t value = train_matrix[row * num_features + col] & 0x03u;
                packed_features |= static_cast<uint8_t>(value << (col * cfg.quantization_bits));
            }
            fout.write(reinterpret_cast<const char*>(&label), sizeof(label));
            fout.write(reinterpret_cast<const char*>(&packed_features), sizeof(packed_features));
        }
    }

    const bool trained = forest.build_model(false, tmp_nml);
    std::filesystem::remove(tmp_nml);
    assert(trained);
    assert(forest.tree_container().trained());
    assert(forest.tree_container().num_trees() == 32u);

    const uint8_t benign_sample[4] = {0u, 1u, 0u, 1u};
    const uint8_t outlier_sample[4] = {3u, 3u, 3u, 3u};

    const float benign_score = forest.decision_function(benign_sample, 4u);
    const float outlier_score = forest.decision_function(outlier_sample, 4u);
    assert(outlier_score < benign_score);
}

} // namespace

int main() {
    test_node_resource_and_node_layout();
    test_quantized_iforest_scoring_order();
    std::cout << "pe_model_engine_tests: PASS\n";
    return 0;
}
