# Does Distractor Position Affect LLM Reasoning?

### An Exploratory Study of Irrelevant Context in Mathematical Reasoning

This project investigates whether the position of irrelevant information within a mathematical reasoning prompt affects the accuracy of a large language model.

## Research Question

Does the position of irrelevant information within a prompt affect LLM reasoning accuracy?

## Experimental Design

I sampled 30 multi-step mathematical reasoning problems from the GSM8K benchmark and evaluated each problem under four conditions:

- **Clean:** no irrelevant information
- **First:** an irrelevant sentence appears at the beginning
- **Middle:** the same sentence is inserted within the problem
- **Last:** the same sentence appears after the relevant information but before the final question

The distractor sentence remained identical across the three distractor conditions for each problem.

This resulted in **120 total experimental prompts**.

## Model

The main experiment used **Qwen2.5-7B-Instruct** with deterministic generation (`do_sample=False`).

Responses were generated using 4-bit quantization with a maximum generation length of 768 tokens.

A preliminary pilot using Qwen2.5-3B-Instruct was excluded from the main analysis because its low clean-condition accuracy created a floor effect.

## Results

| Condition | Correct | Accuracy |
|---|---:|---:|
| Clean | 28/30 | 93.3% |
| Distractor First | 29/30 | 96.7% |
| Distractor Middle | 30/30 | 100.0% |
| Distractor Last | 29/30 | 96.7% |

A Cochran's Q test found no statistically significant difference in accuracy across the four conditions:

**Q = 4.00, p = .261**

The experiment therefore did not provide sufficient evidence that distractor position systematically affected overall reasoning accuracy.

## An Interesting Observation

Although aggregate accuracy changed very little, some individual problems showed different reasoning paths after irrelevant information was introduced.

For example, in one problem the model ignored a relevant piece of information in the clean condition but correctly incorporated it in all three distractor conditions. In another, the model made the same conceptual error in the clean, first, and last conditions but solved the problem correctly when the distractor appeared in the middle.

These cases do not show that irrelevant information improves reasoning. Instead, they raise a broader question about the stability of LLM reasoning under small, semantically irrelevant prompt changes.

## Limitations

This is a small exploratory study using 30 problems and one language model. The model also achieved high baseline accuracy, creating a possible ceiling effect.

Future work could evaluate a larger sample, additional models, more difficult reasoning tasks, and different types of irrelevant information.

## Repository Contents

- [`report.pdf`](report.pdf) — full research report
- [`data.csv`](data.csv) — experimental questions and distractors
- [`results.csv`](results.csv) — raw model outputs and evaluation results
- [`experiment.py`](experiment.py) — experiment and analysis code
- [`requirements.txt`](requirements.txt) — Python dependencies
- [`accuracy_by_distractor_position.png`](accuracy_by_distractor_position.png) — results visualization

## References

Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168.

Qwen Team, et al. (2024). *Qwen2.5 Technical Report*. arXiv:2412.15115.
