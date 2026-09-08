!(import! &self (library lib_import))
!(git-import! "https://github.com/trueagi-io/PLN.git")
!(import! &self (library PLN lib_pln))

; Candidate STVs
(= (STV C3) (stv 0.607143 0.857143))
(= (STV C4) (stv 0.600000 0.800000))
(= (STV C5) (stv 0.607143 0.857143))
(= (STV C8) (stv 0.600000 0.800000))
(= (STV C10) (stv 0.600000 0.800000))
(= (STV C12) (stv 0.600000 0.800000))
(= (STV C15) (stv 0.600000 0.800000))
(= (STV C20) (stv 0.600000 0.800000))
(= (STV C6) (stv 0.500000 0.869561))
(= (STV C9) (stv 0.500000 0.800000))
(= (STV C11) (stv 0.500000 0.800000))
(= (STV C13) (stv 0.500000 0.800000))
(= (STV C16) (stv 0.500000 0.800000))
(= (STV C17) (stv 0.500000 0.869561))
(= (STV C18) (stv 0.500000 0.800000))
(= (STV C19) (stv 0.500000 0.869561))
(= (STV C1) (stv 0.392857 0.857143))
(= (STV C2) (stv 0.400000 0.800000))
(= (STV C7) (stv 0.400000 0.800000))
(= (STV C14) (stv 0.392857 0.857143))
(= (STV GroupX) (stv 0.500000 0.800000))

; Graph KB
(= (kb)
  (
    (Sentence ((Similarity C6 C19) (stv 1.000000 0.800000)) (101))
  )
)

!(PLN.Query (kb) (Similarity C6 C19))

; Deduction
(= (deduction_kb)
  (
    (Sentence ((Inheritance C1 C2) (stv 0.900000 0.800000)) (201))
    (Sentence ((Inheritance C2 C3) (stv 0.800000 0.800000)) (202))
  )
)
!(PLN.Query (deduction_kb) (Inheritance C1 C3))

; Induction
(= (induction_kb)
  (
    (Sentence ((Inheritance GroupX C1) (stv 0.900000 0.800000)) (301))
    (Sentence ((Inheritance GroupX C2) (stv 0.800000 0.800000)) (302))
  )
)
!(PLN.Query (induction_kb) (Inheritance C1 C2))

; Abduction
(= (abduction_kb)
  (
    (Sentence ((Inheritance C1 C3) (stv 0.600000 0.800000)) (401))
    (Sentence ((Inheritance C2 C3) (stv 0.600000 0.800000)) (402))
  )
)
!(PLN.Query (abduction_kb) (Inheritance C1 C2))

; Revision
(= (revision_kb)
  (
    (Sentence ((Inheritance C1 C2) (stv 0.700000 0.600000)) (501))
    (Sentence ((Inheritance C1 C2) (stv 0.900000 0.800000)) (502))
  )
)
!(PLN.Query (revision_kb) (Inheritance C1 C2))
