from itertools import product

truth_table = [
    (A, B, C, A ^ B ^ C)
    for A, B, C in product((0, 1), repeat=3)
]

ROW_WEIGHTS = [1.0] * len(truth_table)

# Candidate programs.
candidates = {
    "C1":  "(AND A (OR B C))",
    "C2":  "(OR (AND A B) C)",
    "C3":  "(NOT (AND A (OR B C)))",
    "C4":  "(OR (NOT A) (AND B C))",
    "C5":  "(AND (OR A B) (NOT C))",
    "C6":  "(OR (AND A (NOT B)) (AND B C))",
    "C7":  "(AND (NOT A) (OR B (NOT C)))",
    "C8":  "(OR (NOT (AND A B)) C)",
    "C9":  "(AND (OR A (NOT B)) (OR B C))",
    "C10": "(NOT (OR (AND A C) B))",
    "C11": "(OR (AND A B) (AND (NOT A) C))",
    "C12": "(AND (OR A C) (OR (NOT B) C))",
    "C13": "(OR (AND (NOT A) B) (AND A C))",
    "C14": "(NOT (AND (OR A B) (NOT C)))",
    "C15": "(AND (NOT (OR A C)) (OR B C))",
    "C16": "(OR (AND A (NOT C)) (AND (NOT B) C))",
    "C17": "(AND (OR (NOT A) B) (NOT (AND B C)))",
    "C18": "(OR (NOT (OR A B)) (AND A C))",
    "C19": "(AND (OR A (AND B C)) (OR (NOT B) C))",
    "C20": "(OR (AND A (NOT B)) (AND (OR B C) (NOT A)))",
}


def tokenize(expression):
    return expression.replace("(", " ( ").replace(")", " ) ").split()


def parse(expressions):
    expr = expressions.pop(0)

    if expr == "(":
        operator = expressions.pop(0)
        args = []

        while expressions[0] != ")":
            args.append(parse(expressions))

        expressions.pop(0)
        return (operator, *args)
    
    return expr


def evaluate(node, values):
    if isinstance(node, str):
        return values[node]

    operator = node[0]
    args = node[1:]

    if operator == "AND":
        result = all(evaluate(arg, values) for arg in args)
        return int(result)
    
    if operator == "OR":
        result = any(evaluate(arg, values) for arg in args)
        return int(result)

    if operator == "NOT":
        result = evaluate(args[0], values)
        return int(not result)
    
    raise ValueError("The operator mentioned in candidate dict must match with AND, OR, NOT")


def evaluate_expressions(expression, A, B, C):
    expr = tokenize(expression)
    parsed_format = parse(expr)

    values = {
        "A": A,
        "B": B,
        "C": C
    }
    return evaluate(parsed_format, values)


def evaluate_all_candidates(sort_result=False):
    results = {}
    for name, expr in candidates.items():
        outputs = []
        correct = 0

        for A, B, C, target in truth_table:
            predicted = evaluate_expressions(expr, A, B, C)

            outputs.append(predicted)
            if predicted == target:
                correct += 1

        incorrect = len(truth_table) - correct
        score = correct / len(truth_table)
        results[name] = {
            "outputs": tuple(outputs),
            "correct": correct,
            "incorrect": incorrect,
            "score": score
        }

    if sort_result:
        results = dict(
            sorted(
                results.items(),
                key=lambda item: item[1]["score"],
                reverse=True,
            )
        )

    return results



def calculate_weighted_evidence(results, weights=ROW_WEIGHTS):
    weighted_results = {}

    targets = [row[3] for row in truth_table]
    for name, result in results.items():
        outputs = result["outputs"]

        weighted_success = 0.0
        weighted_failure = 0.0

        for predicted, target, weight in zip(outputs, targets, weights):
            if predicted == target:
                weighted_success += weight
            else:
                weighted_failure += weight
        weighted_results[name] = {
            "success": weighted_success,
            "failure": weighted_failure,
        }

    return weighted_results