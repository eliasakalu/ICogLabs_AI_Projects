class BetaNode:
    def __init__(self, name, alpha_prior=1, beta_prior=1) -> None:
        self.name = name
        self.alpha = float(alpha_prior)
        self.beta = float(beta_prior)

    def update_evidence(self, success, failure):
        self.alpha += success
        self.beta += failure

    @property
    def expected_value(self):
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    def __repr__(self) -> str:
        return f"BetaNode({self.name} | Beta({self.alpha}, {self.beta}) | E[p]={self.expected_value:.3f})"

    def to_stv(self, k=2.0):
        evidence = max(0.0, self.alpha + self.beta - 2.0)
        strength = self.expected_value
        confidence = evidence / (evidence + k) if (evidence + k) > 0 else 0.0

        return strength, confidence