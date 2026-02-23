#pragma once

#include <filesystem>
#include <random>
#include <vector>

#include "if_base.h"
#include "if_config.h"
#include "if_scaler_transform.h"
#include "if_node_resource.h"
#include "if_model.h"

namespace eml {

	class If_tree {
	private:
		IsoTree tree_{};

	public:
		void set_resource(const If_node_resource& resource) {
			tree_.set_resource(resource);
		}

		const If_node_resource& resource() const {
			return tree_.resource();
		}

		bool train(const uint8_t* matrix,
				   size_t num_samples,
				   uint16_t num_features,
				   const std::vector<uint32_t>& sampled_indices,
				   uint16_t max_depth,
				   uint32_t max_nodes_per_tree,
				   std::mt19937& rng) {
			return tree_.train(matrix,
							   num_samples,
							   num_features,
							   sampled_indices,
							   max_depth,
							   max_nodes_per_tree,
							   rng);
		}

		float path_length(const uint8_t* quantized_features, uint16_t num_features) const {
			return tree_.path_length(quantized_features, num_features);
		}

		size_t node_count() const { return tree_.node_count(); }
		uint16_t depth() const { return tree_.depth(); }
		bool loaded() const { return tree_.loaded(); }

		const IsoTree& impl() const { return tree_; }
		IsoTree& impl() { return tree_; }
	};

	class If_tree_container {
	private:
		QuantizedIsolationForest forest_{};

	public:
		bool train(const uint8_t* matrix,
				   size_t num_samples,
				   uint16_t num_features,
				   const If_config& cfg) {
			return forest_.train(matrix, num_samples, num_features, cfg);
		}

		float score_samples(const uint8_t* quantized_features, uint16_t num_features) const {
			return forest_.score_samples(quantized_features, num_features);
		}

		float decision_function(const uint8_t* quantized_features, uint16_t num_features) const {
			return forest_.decision_function(quantized_features, num_features);
		}

		bool is_anomaly(const uint8_t* quantized_features,
						uint16_t num_features,
						float threshold) const {
			return forest_.is_anomaly(quantized_features, num_features, threshold);
		}

		bool save_model_binary(const std::filesystem::path& model_path) const {
			return forest_.save_model_binary(model_path);
		}

		bool load_model_binary(const std::filesystem::path& model_path) {
			return forest_.load_model_binary(model_path);
		}

		size_t num_trees() const { return forest_.num_trees(); }
		uint32_t samples_per_tree() const { return forest_.samples_per_tree(); }
		bool trained() const { return forest_.trained(); }
		float threshold_offset() const { return forest_.threshold_offset(); }

		const QuantizedIsolationForest& impl() const { return forest_; }
		QuantizedIsolationForest& impl() { return forest_; }
	};

} // namespace eml
