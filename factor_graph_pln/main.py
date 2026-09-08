from parity_tools import (
    evaluate_all_candidates, calculate_weighted_evidence, truth_table, ROW_WEIGHTS,
)

from factor_inference import (
    build_correlation_edges,
    create_beta_nodes,
    build_factor_graph,
    run_belief_propagation,
    rank_candidates,
)

from visualizer import visualize_factor_graph


CORR_THRESHOLD = 0.8


def relation_stv(results, source, target, rows):
    """Count target correctness on rows where the source is correct."""
    success = failure = 0.0
    for index in rows:
        expected = truth_table[index][3]
        source_correct = source == "GroupX" or results[source]["outputs"][index] == expected
        if source_correct:
            if results[target]["outputs"][index] == expected:
                success += ROW_WEIGHTS[index]
            else:
                failure += ROW_WEIGHTS[index]
    evidence = success + failure
    # Use the same Beta(1,1) prior and confidence mapping as candidate nodes.
    return f"(stv {(success + 1) / (evidence + 2):.6f} {evidence / (evidence + 2):.6f})"


def write_pln_stvs_and_relations(
    final_beta_nodes,
    edges,
    output_file="./pln_rules.metta"
):
    results = evaluate_all_candidates()
    rows = range(len(truth_table))
    required = {"C1", "C2", "C3"}
    if not required.issubset(final_beta_nodes):
        raise ValueError("The PLN rule examples require candidates C1, C2, and C3")

    with open(output_file, "w") as f:
        # PLN imports
        f.write("""!(import! &self (library lib_import))
        !(git-import! "https://github.com/trueagi-io/PLN.git")
        !(import! &self (library PLN lib_pln))

        """)

        # Candidate STVs
        f.write("; Candidate STVs\n")

        for name, node in final_beta_nodes.items():
            strength, confidence = node.to_stv()

            f.write(
                f"(= (STV {name}) "
                f"(stv {strength:.6f} {confidence:.6f}))\n"
            )

        # GroupX denotes all evaluated input rows.
        total_weight = sum(ROW_WEIGHTS)
        f.write(
            f"(= (STV GroupX) (stv 1.000000 "
            f"{total_weight / (total_weight + 2):.6f}))\n\n"
        )

        # Graph KB
        f.write("; Graph KB\n")
        f.write("(= (kb)\n  (\n")

        positive_edges = []

        for stamp, edge in enumerate(edges, start=101):
            c1 = edge["source"]
            c2 = edge["target"]
            corr = edge["correlation"]

            if corr > 0:
                positive_edges.append(edge)

                f.write(
                    f"    (Sentence ((Similarity {c1} {c2}) "
                    f"(stv {corr:.6f} 0.800000)) "
                    f"({stamp}))\n"
                )

        f.write("  )\n)\n\n")

        if positive_edges:
            edge = positive_edges[0]
            c1 = edge["source"]
            c2 = edge["target"]

            f.write(
                f"!(PLN.Query (kb) "
                f"(Similarity {c1} {c2}))\n\n"
            )
        else:
            f.write(
                "; No positive Similarity edge "
                "available at current threshold\n\n"
            )

        # Relation evidence comes from correctness on the input rows.
        examples = (
            ("Deduction", (("C1", "C2"), ("C2", "C3")), ("C1", "C3"), 201),
            ("Induction", (("GroupX", "C1"), ("GroupX", "C2")), ("C1", "C2"), 301),
            ("Abduction", (("C1", "C3"), ("C2", "C3")), ("C1", "C2"), 401),
            ("Revision", (("C1", "C2"), ("C1", "C2")), ("C1", "C2"), 501),
        )
        f.write("; Inheritance X Y estimates Y correctness given X correctness.\n")
        f.write("; Candidate STVs above retain the approximate graph beliefs.\n")
        for label, premises, query, first_stamp in examples:
            kb = label.lower() + "_kb"
            f.write(f"\n; {label}\n(= ({kb})\n  (\n")
            for offset, (source, target) in enumerate(premises):
                # Revision uses disjoint rows, not two copies of the same evidence.
                evidence_rows = rows[offset::2] if label == "Revision" else rows
                stv = relation_stv(results, source, target, evidence_rows)
                f.write(
                    f"    (Sentence ((Inheritance {source} {target}) {stv}) "
                    f"({first_stamp + offset}))\n"
                )
            f.write(f"  )\n)\n!(PLN.Query ({kb}) (Inheritance {query[0]} {query[1]}))\n")


def main():
    candidates_result = evaluate_all_candidates(sort_result=True)

    edges = build_correlation_edges(
        candidates_result,
        threshold=CORR_THRESHOLD
    )

    weighted_evidence = calculate_weighted_evidence(candidates_result)

    beta_nodes = create_beta_nodes(weighted_evidence)

    final_beta_nodes = run_belief_propagation(
        beta_nodes,
        weighted_evidence,
        edges
    )

    ranking = rank_candidates(final_beta_nodes)

    print("\n=== FINAL CANDIDATE RANKING ===")

    for rank, (name, node) in enumerate(ranking, start=1):
        strength, confidence = node.to_stv()

        print(
            f"{rank:2}. {name} | "
            f"belief={node.expected_value:.3f} | "
            f"STV=({strength:.3f}, {confidence:.3f})"
        )

    write_pln_stvs_and_relations(final_beta_nodes, edges)
    print("\nGenerated pln_rules.metta")

    factor_graph = build_factor_graph(final_beta_nodes, edges)

    visualize_factor_graph(
        factor_graph,
        title="Parity-3 Final Factor Graph"
    )


if __name__ == "__main__":
    main()
