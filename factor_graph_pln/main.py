from parity_tools import evaluate_all_candidates, calculate_weighted_evidence

from factor_inference import (
    build_correlation_edges,
    create_beta_nodes,
    build_factor_graph,
    run_belief_propagation,
    rank_candidates,
)

from visualizer import visualize_factor_graph


CORR_THRESHOLD = 0.8


def write_pln_stvs_and_relations(
    final_beta_nodes,
    edges,
    output_file="./pln_rules.metta"
):
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

        f.write(
            "(= (STV GroupX) "
            "(stv 0.500000 0.800000))\n\n"
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

        # Deduction
        f.write("""; Deduction
(= (deduction_kb)
  (
    (Sentence ((Inheritance C1 C2) (stv 0.900000 0.800000)) (201))
    (Sentence ((Inheritance C2 C3) (stv 0.800000 0.800000)) (202))
  )
)
!(PLN.Query (deduction_kb) (Inheritance C1 C3))

""")

        # Induction
        f.write("""; Induction
(= (induction_kb)
  (
    (Sentence ((Inheritance GroupX C1) (stv 0.900000 0.800000)) (301))
    (Sentence ((Inheritance GroupX C2) (stv 0.800000 0.800000)) (302))
  )
)
!(PLN.Query (induction_kb) (Inheritance C1 C2))

""")

        # Abduction
        f.write("""; Abduction
(= (abduction_kb)
  (
    (Sentence ((Inheritance C1 C3) (stv 0.600000 0.800000)) (401))
    (Sentence ((Inheritance C2 C3) (stv 0.600000 0.800000)) (402))
  )
)
!(PLN.Query (abduction_kb) (Inheritance C1 C2))

""")

        # Revision
        f.write("""; Revision
(= (revision_kb)
  (
    (Sentence ((Inheritance C1 C2) (stv 0.700000 0.600000)) (501))
    (Sentence ((Inheritance C1 C2) (stv 0.900000 0.800000)) (502))
  )
)
!(PLN.Query (revision_kb) (Inheritance C1 C2))
""")


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
