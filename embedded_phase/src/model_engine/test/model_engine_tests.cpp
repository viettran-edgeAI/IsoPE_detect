#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "model_engine.hpp"

namespace {

    void test_node_resource_and_node_layout() {
        eml::If_node_resource resource;
        const bool ok = resource.set_bits(3, 6, 16, 15, 5);
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

        eml::QuantizedIsolationForest forest;
        const bool trained = forest.train(train_matrix.data(), 8u, 4u, cfg);
        assert(trained);
        assert(forest.trained());
        assert(forest.num_trees() == 32u);

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
