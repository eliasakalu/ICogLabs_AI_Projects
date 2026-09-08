import argparse
import subprocess
from itertools import permutations
from pathlib import Path

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


def select_rule_examples(results, beta_nodes):
    """Choose supported premises from the current candidates, with stable ties."""
    names = sorted(beta_nodes)
    rows = range(len(truth_table))
    if set(results) != set(beta_nodes):
        raise ValueError("Candidate results and Beta nodes must have the same names")
    if len(ROW_WEIGHTS) != len(rows) or any(w < 0 for w in ROW_WEIGHTS):
        raise ValueError("Provide one nonnegative weight per truth-table row")

    def support(source, target, indices):
        source_rows = [i for i in indices
                       if results[source]["outputs"][i] == truth_table[i][3]]
        evidence = sum(ROW_WEIGHTS[i] for i in source_rows)
        matches = sum(ROW_WEIGHTS[i] for i in source_rows
                      if results[target]["outputs"][i] == truth_table[i][3])
        return evidence, matches

    full = {(a, b): support(a, b, rows) for a, b in permutations(names, 2)}
    selected = {}

    def consider(label, premises, query, supports):
        if any(evidence <= 0 or matches <= 0 for evidence, matches in supports):
            return
        score = (min(e for e, _ in supports), sum(m for _, m in supports))
        if label not in selected or score > selected[label][0]:
            selected[label] = (score, premises, query)

    for a, b, c in permutations(names, 3):
        consider("Deduction", ((a, b), (b, c)), (a, c), (full[a, b], full[b, c]))
        consider("Induction", ((c, a), (c, b)), (a, b), (full[c, a], full[c, b]))
        consider("Abduction", ((a, c), (b, c)), (a, b), (full[a, c], full[b, c]))
    for a, b in permutations(names, 2):
        consider("Revision", ((a, b), (a, b)), (a, b),
                 (support(a, b, rows[::2]), support(a, b, rows[1::2])))

    examples = []
    for index, label in enumerate(("Deduction", "Induction", "Abduction", "Revision"), 2):
        if label in selected:
            _, premises, query = selected[label]
            examples.append((label, premises, query, index * 100 + 1))
        else:
            print(f"{label}: skipped because no supported example is available")
    return examples


def write_pln_stvs_and_relations(
    final_beta_nodes,
    edges,
    output_file="./pln_rules.metta",
    results=None,
):
    if results is None:
        results = evaluate_all_candidates()
    rows = range(len(truth_table))
    examples = select_rule_examples(results, final_beta_nodes)

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
        f.write("; Inheritance X Y estimates Y correctness given X correctness.\n")
        f.write("; Candidate STVs above retain the approximate graph beliefs.\n")
        for label, premises, query, first_stamp in examples:
            print(f"{label}: {premises[0]} + {premises[1]} -> {query}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-pln", action="store_true", help="Also execute the generated PLN queries")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent

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

    write_pln_stvs_and_relations(
        final_beta_nodes, edges, project_dir / "pln_rules.metta", candidates_result
    )
    print("\nGenerated pln_rules.metta")

    if args.run_pln:
        petta_dir = project_dir / "PeTTa"
        if not (petta_dir / "run.sh").is_file():
            raise FileNotFoundError("Install PeTTa in factor_graph_pln/PeTTa to run PLN")
        subprocess.run(["sh", "run.sh", "../pln_rules.metta"], cwd=petta_dir, check=True)

    factor_graph = build_factor_graph(final_beta_nodes, edges)

    visualize_factor_graph(
        factor_graph,
        title="Parity-3 Final Factor Graph"
    )


if __name__ == "__main__":
    main()
