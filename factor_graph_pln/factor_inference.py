import networkx as nx
from itertools import combinations

from betanode import BetaNode


def build_correlation_edges(results, threshold=0.8):
    groups = {}

    # Group programs that produce the same outputs.
    for name, result in results.items():
        signature = tuple(result["outputs"])
        groups.setdefault(signature, []).append(name)

    edges = []

    # Same outputs have correlation +1
    for names in groups.values():
        for c1, c2 in combinations(names, 2):
            if 1.0 >= threshold:
                edges.append({
                    "source": c1,
                    "target": c2,
                    "correlation": 1.0,
                })

    checked = set()

    # Opposite outputs have correlation -1.
    for signature, names in groups.items():
        complement = tuple(1 - bit for bit in signature)

        if complement not in groups:
            continue

        pair_key = frozenset((signature, complement))
        if pair_key in checked:
            continue

        checked.add(pair_key)

        if 1.0 >= threshold:
            for c1 in names:
                for c2 in groups[complement]:
                    edges.append({
                        "source": c1,
                        "target": c2,
                        "correlation": -1.0,
                    })

    return edges


def create_beta_nodes(weighted_evidence):
    beta_node = {}

    for name, evidence in weighted_evidence.items():
        node = BetaNode(name, alpha_prior=1, beta_prior=1)
        node.update_evidence(
            success=evidence["success"],
            failure=evidence["failure"]
        )

        beta_node[name] = node

    return beta_node


def build_factor_graph(beta_nodes, edges):
    graph = nx.Graph()

    for name, node in beta_nodes.items():
        graph.add_node(
            name,
            node_type="variable",
            alpha=node.alpha,
            beta=node.beta,
            belief=node.expected_value,
        )

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        corr = edge["correlation"]

        factor_name = f"f_{source}_{target}"

        graph.add_node(
            factor_name,
            node_type="factor",
            correlation=corr,
            strength=abs(corr),
            relation="similar" if corr > 0 else "opposite",
        )

        graph.add_edge(source, factor_name)
        graph.add_edge(factor_name, target)

    return graph


def beta_message(successes, failures, correlation):
    strength = abs(correlation)

    if correlation >= 0:
        return {
            "success": strength * successes,
            "failure": strength * failures,
        }

    # For opposite outputs, swap successes and failures.
    return {
        "success": strength * failures,
        "failure": strength * successes,
    }


def run_belief_propagation(
    beta_nodes, weighted_evidence, edges, max_iterations=100,
    tolerance=1e-4, damping=0.5, coupling=0.5
):
    """Share part of each node's evidence with its neighbors.

    This approximates Beta updates; it is not exact sum-product BP.
    Coupling reduces message strength, and damping smooths updates.
    """
    if not 0 <= damping < 1 or not 0 <= coupling < 1:
        raise ValueError("damping and coupling must be in [0, 1)")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("iterations and tolerance must be positive")

    neighbors = {name: {} for name in beta_nodes}
    for edge in edges:
        src, dst, corr = edge["source"], edge["target"], edge["correlation"]
        neighbors[src][dst] = corr
        neighbors[dst][src] = corr

    messages = {(src, dst): (0.0, 0.0)
                for src in neighbors for dst in neighbors[src]}
    for iteration in range(max_iterations):
        updated = {}
        max_change = 0.0
        for (src, dst), previous in messages.items():
            # Do not send the destination's message straight back to it.
            others = [name for name in neighbors[src] if name != dst]
            denominator = 1 + sum(abs(neighbors[src][name]) for name in others)
            success = weighted_evidence[src]["success"]
            failure = weighted_evidence[src]["failure"]
            success += sum(messages[name, src][0] for name in others)
            failure += sum(messages[name, src][1] for name in others)
            raw = beta_message(success, failure, neighbors[src][dst])
            target = (coupling * raw["success"] / denominator,
                      coupling * raw["failure"] / denominator)
            updated[src, dst] = tuple(
                damping * old + (1 - damping) * new
                for old, new in zip(previous, target)
            )
            max_change = max(max_change, *(abs(new - old) for new, old
                                          in zip(updated[src, dst], previous)))
        messages = updated
        print(f"Iteration {iteration + 1}: message_change={max_change:.6f}")
        if max_change < tolerance:
            print("Converged")
            break
    else:
        print("Iteration limit reached; messages have not converged")

    final_nodes = {}
    for name, initial in beta_nodes.items():
        node = BetaNode(name, alpha_prior=initial.alpha, beta_prior=initial.beta)
        node.update_evidence(
            success=sum(messages[src, name][0] for src in neighbors[name]),
            failure=sum(messages[src, name][1] for src in neighbors[name]),
        )
        final_nodes[name] = node
    return final_nodes


def rank_candidates(final_beta_nodes):
    return sorted(
        final_beta_nodes.items(),
        key=lambda item: item[1].expected_value,
        reverse=True,
    )
