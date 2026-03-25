/**
 * C++ MCTS tree for draft selection.
 *
 * Handles UCB selection, expansion, and backpropagation in C++.
 * Neural network forward passes stay in Python (called via callback).
 *
 * This eliminates Python dict/object overhead in the inner MCTS loop,
 * which accounts for ~40% of simulation time.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>

namespace py = pybind11;

static constexpr int NUM_HEROES = 90;
static constexpr float C_PUCT_DEFAULT = 2.0f;

struct MCTSNode {
    int action;         // hero index that led to this node (-1 for root)
    int parent_idx;     // index in nodes vector (-1 for root)
    int visit_count;
    float value_sum;
    float prior;
    bool is_expanded;

    // Children: action -> node index in the flat vector
    // Use fixed-size array for speed (90 possible actions)
    int children[NUM_HEROES];  // -1 = no child
    int num_children;

    MCTSNode() : action(-1), parent_idx(-1), visit_count(0), value_sum(0.0f),
                 prior(0.0f), is_expanded(false), num_children(0) {
        std::memset(children, -1, sizeof(children));
    }

    float q_value() const {
        return visit_count == 0 ? 0.0f : value_sum / visit_count;
    }

    float ucb_score(int parent_visits, float c_puct) const {
        float exploration = c_puct * prior * std::sqrt(static_cast<float>(parent_visits))
                            / (1.0f + visit_count);
        return q_value() + exploration;
    }
};


class MCTSTree {
public:
    std::vector<MCTSNode> nodes;

    MCTSTree() {
        nodes.reserve(8192);
        // Create root
        nodes.emplace_back();
    }

    int root() const { return 0; }

    int select_child_ucb(int node_idx, float c_puct) const {
        const MCTSNode& node = nodes[node_idx];
        int best_child = -1;
        float best_score = -1e9f;

        for (int a = 0; a < NUM_HEROES; ++a) {
            int child_idx = node.children[a];
            if (child_idx < 0) continue;
            float score = nodes[child_idx].ucb_score(node.visit_count, c_puct);
            if (score > best_score) {
                best_score = score;
                best_child = child_idx;
            }
        }
        return best_child;
    }

    int get_child_action(int child_idx) const {
        return nodes[child_idx].action;
    }

    bool is_expanded(int node_idx) const {
        return nodes[node_idx].is_expanded;
    }

    // Expand node with priors for valid actions
    void expand(int node_idx, py::array_t<float> priors_np, py::array_t<float> valid_np) {
        auto priors = priors_np.unchecked<1>();
        auto valid = valid_np.unchecked<1>();

        MCTSNode& node = nodes[node_idx];
        node.is_expanded = true;
        node.num_children = 0;

        for (int a = 0; a < NUM_HEROES; ++a) {
            if (valid(a) > 0.0f && priors(a) > 0.0f) {
                int child_idx = static_cast<int>(nodes.size());
                nodes.emplace_back();
                MCTSNode& child = nodes.back();
                child.action = a;
                child.parent_idx = node_idx;
                child.prior = priors(a);
                node.children[a] = child_idx;
                node.num_children++;
            }
        }
    }

    // Backpropagate value from node to root
    void backprop(int node_idx, float value) {
        int idx = node_idx;
        while (idx >= 0) {
            nodes[idx].visit_count++;
            nodes[idx].value_sum += value;
            idx = nodes[idx].parent_idx;
        }
    }

    // Get visit count distribution over actions from root
    py::array_t<float> get_visit_distribution() const {
        auto result = py::array_t<float>(NUM_HEROES);
        auto buf = result.mutable_unchecked<1>();

        float total = 0.0f;
        for (int a = 0; a < NUM_HEROES; ++a) {
            int child_idx = nodes[0].children[a];
            float visits = (child_idx >= 0) ? static_cast<float>(nodes[child_idx].visit_count) : 0.0f;
            buf(a) = visits;
            total += visits;
        }

        if (total > 0.0f) {
            for (int a = 0; a < NUM_HEROES; ++a) {
                buf(a) /= total;
            }
        }

        return result;
    }

    int root_visit_count() const { return nodes[0].visit_count; }
    int num_nodes() const { return static_cast<int>(nodes.size()); }
};


/**
 * Run MCTS search. The network and GD calls are Python callbacks.
 *
 * network_predict_fn(state_np) -> (priors_np, value)
 * gd_sample_fn(state_np, valid_np) -> action_int
 * apply_action_fn(state_dict, action, team, action_type) -> new_state_dict
 *
 * We pass state as a dict-like object to Python for manipulation,
 * since the DraftState logic is complex and stays in Python.
 */
py::array_t<float> mcts_search(
    py::object root_state,         // DraftState Python object
    py::object network_predict_fn, // callable(state) -> (priors, value)
    py::object gd_sample_fn,       // callable(state) -> action
    int our_team,
    int num_simulations,
    float c_puct = C_PUCT_DEFAULT
) {
    MCTSTree tree;

    // Expand root
    {
        py::tuple result = network_predict_fn(root_state).cast<py::tuple>();
        py::array_t<float> priors = result[0].cast<py::array_t<float>>();
        // float root_value = result[1].cast<float>();  // not used for root

        py::array_t<float> valid = root_state.attr("valid_mask_np")().cast<py::array_t<float>>();

        // Normalize priors over valid actions
        auto p = priors.mutable_unchecked<1>();
        auto v = valid.unchecked<1>();
        float sum = 0.0f;
        for (int a = 0; a < NUM_HEROES; ++a) {
            p(a) *= v(a);
            sum += p(a);
        }
        if (sum > 0.0f) {
            for (int a = 0; a < NUM_HEROES; ++a) p(a) /= sum;
        }

        tree.expand(tree.root(), priors, valid);
    }

    // Get draft order for determining whose turn it is
    py::list draft_order = py::module_::import("train_draft_policy").attr("DRAFT_ORDER").cast<py::list>();

    for (int sim = 0; sim < num_simulations; ++sim) {
        // Clone root state for this simulation
        py::object scratch = root_state.attr("clone")();
        int node_idx = tree.root();

        // Selection: traverse tree using UCB
        while (tree.is_expanded(node_idx) && !scratch.attr("is_terminal")().cast<bool>()) {
            int step = scratch.attr("step").cast<int>();
            py::tuple step_info = draft_order[step].cast<py::tuple>();
            int step_team = step_info[0].cast<int>();
            py::str action_type = step_info[1].cast<py::str>();

            if (step_team == our_team) {
                // Our turn: UCB selection
                int child_idx = tree.select_child_ucb(node_idx, c_puct);
                if (child_idx < 0) break;
                int action = tree.get_child_action(child_idx);
                scratch.attr("apply_action")(action, step_team, action_type);
                node_idx = child_idx;
            } else {
                // Opponent: sample from GD (pass-through, not in tree)
                int opp_action = gd_sample_fn(scratch).cast<int>();
                scratch.attr("apply_action")(opp_action, step_team, action_type);
                // Stay at same tree node (opponent actions are pass-through)
            }
        }

        float value;
        if (scratch.attr("is_terminal")().cast<bool>()) {
            // Terminal: get value from Python (WP evaluation)
            value = py::module_::import("train_draft_policy")
                .attr("_evaluate_wp_for_mcts")(scratch).cast<float>();
        } else {
            int step = scratch.attr("step").cast<int>();
            py::tuple step_info = draft_order[step].cast<py::tuple>();
            int step_team = step_info[0].cast<int>();

            if (!tree.is_expanded(node_idx) && step_team == our_team) {
                // Expand leaf
                py::tuple result = network_predict_fn(scratch).cast<py::tuple>();
                py::array_t<float> priors = result[0].cast<py::array_t<float>>();
                value = result[1].cast<float>();

                py::array_t<float> valid = scratch.attr("valid_mask_np")().cast<py::array_t<float>>();

                auto p = priors.mutable_unchecked<1>();
                auto v = valid.unchecked<1>();
                float sum = 0.0f;
                for (int a = 0; a < NUM_HEROES; ++a) {
                    p(a) *= v(a);
                    sum += p(a);
                }
                if (sum > 0.0f) {
                    for (int a = 0; a < NUM_HEROES; ++a) p(a) /= sum;
                }

                tree.expand(node_idx, priors, valid);
            } else {
                // At opponent's turn or already expanded: just get value
                py::tuple result = network_predict_fn(scratch).cast<py::tuple>();
                value = result[1].cast<float>();
            }
        }

        // Backpropagation
        tree.backprop(node_idx, value);
    }

    return tree.get_visit_distribution();
}


PYBIND11_MODULE(mcts_core, m) {
    m.doc() = "C++ MCTS core for draft selection";

    py::class_<MCTSTree>(m, "MCTSTree")
        .def(py::init<>())
        .def("root_visit_count", &MCTSTree::root_visit_count)
        .def("num_nodes", &MCTSTree::num_nodes)
        .def("get_visit_distribution", &MCTSTree::get_visit_distribution);

    m.def("mcts_search", &mcts_search,
          py::arg("root_state"),
          py::arg("network_predict_fn"),
          py::arg("gd_sample_fn"),
          py::arg("our_team"),
          py::arg("num_simulations"),
          py::arg("c_puct") = C_PUCT_DEFAULT,
          "Run MCTS search from root_state. Returns visit distribution.");
}
