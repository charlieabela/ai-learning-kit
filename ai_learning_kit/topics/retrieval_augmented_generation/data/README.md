# Data for the retrieval-augmented-generation topic

`medquad_subset.jsonl.gz`: a subset of MedQuAD (Ben Abacha, A. and Demner-Fushman, D. (2019). A question-entailment
approach to question answering. BMC Bioinformatics, 20, 511), taken from commit 577bd37 of
https://github.com/abachaa/MedQuAD and released under the Creative Commons Attribution 4.0 International licence
(https://creativecommons.org/licenses/by/4.0/). Changes: only answered question-answer pairs are kept; the 795 answers
whose focus (or a listed synonym) matches a Hetionet disease name, plus 2,000 answers about other conditions drawn
with seed 5111 as distractors; whitespace is normalised; the field `hetionet_disease` is added. The answers are NIH
consumer-health text (GARD, GHR, NIDDK, NINDS, MedlinePlus Health Topics, NIHSeniorHealth, CancerGov, NHLBI, CDC).

The formulary in tools.py was written for teaching from the knowledge-graphs drug-interaction example: simplified, not prescribing guidance.
