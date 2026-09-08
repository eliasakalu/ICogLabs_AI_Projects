# Factor Graph and PLN Parity-3

Evaluates 20 Boolean candidates using Beta beliefs and demonstrates four PLN rules.

## Run

From `factor_graph_pln/`:

```bash
pip install -r requirements.txt
python main.py
```
In other terminal or after you close the Graph Matplot GUI
```bash
cd PeTTa
sh run.sh ../pln_rules.metta
```

PeTTa requires SWI-Prolog >= 9.3.x;

## Model and results

Candidates start with Beta(1,1); correct/incorrect predictions add evidence.
Correlation links pass discounted evidence, swapping it for negative links.
Belief is `alpha / (alpha + beta)`. Propagation is approximate and shared
observations can inflate confidence. One supported example per rule is selected automatically.
Premise STVs come from weighted correctness counts; revision uses separate row subsets.

- `output.log`: rankings; C3 and C5 lead with belief ≈ 0.607.
- `pln_query_results.log`: similarity, deduction, induction, abduction, revision.
- `Figure_2.png`: graph diagram; `pln_rules.metta`: generated STVs and queries.
