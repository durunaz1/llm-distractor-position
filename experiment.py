import re
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from statsmodels.stats.contingency_tables import cochrans_q


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "data.csv"
RESULTS_PATH = "results.csv"

MAX_NEW_TOKENS = 768


# -----------------------------
# Prompt construction
# -----------------------------

def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def insert_middle(question, distractor):
    sentences = split_sentences(question)
    middle_index = len(sentences) // 2

    return " ".join(
        sentences[:middle_index]
        + [distractor]
        + sentences[middle_index:]
    )


def insert_before_final_question(question, distractor):
    sentences = split_sentences(question)

    if len(sentences) < 2:
        return f"{distractor} {question}"

    return " ".join(
        sentences[:-1]
        + [distractor]
        + [sentences[-1]]
    )


def build_experiment(data):
    rows = []

    for _, row in data.iterrows():
        question = row["question"].strip()
        distractor = row["distractor"].strip()

        conditions = {
            "clean": question,
            "first": f"{distractor} {question}",
            "middle": insert_middle(question, distractor),
            "last": insert_before_final_question(
                question,
                distractor
            ),
        }

        for condition, problem_text in conditions.items():
            prompt = (
                f"{problem_text}\n\n"
                "Solve the problem step by step. "
                "At the end, write FINAL: followed by only "
                "the numerical answer."
            )

            rows.append({
                "question_id": row["question_id"],
                "condition": condition,
                "prompt": prompt,
                "final_answer": row["final_answer"],
                "distractor": distractor,
            })

    return pd.DataFrame(rows)


# -----------------------------
# Model setup
# -----------------------------

def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    return tokenizer, model


# -----------------------------
# Inference
# -----------------------------

def ask_model(prompt, tokenizer, model):
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# -----------------------------
# Scoring
# -----------------------------

def extract_final_number(text):
    match = re.search(
        r"\*{0,2}FINAL:\*{0,2}\s*(-?\d+(?:\.\d+)?)",
        str(text).replace(",", ""),
        re.IGNORECASE,
    )

    return match.group(1) if match else None


def score_output(model_output, final_answer):
    predicted = extract_final_number(model_output)

    if predicted is None:
        return None

    try:
        return int(
            abs(
                float(predicted)
                - float(final_answer)
            ) < 1e-6
        )
    except ValueError:
        return None


# -----------------------------
# Main experiment
# -----------------------------

def main():
    data = pd.read_csv(DATA_PATH)

    experiment_df = build_experiment(data)

    print(
        f"Created {len(experiment_df)} prompts "
        f"from {len(data)} problems."
    )

    tokenizer, model = load_model()

    results = []

    for i, row in experiment_df.iterrows():
        output = ask_model(
            row["prompt"],
            tokenizer,
            model,
        )

        results.append({
            "question_id": row["question_id"],
            "condition": row["condition"],
            "model_output": output,
            "final_answer": row["final_answer"],
            "correct": score_output(
                output,
                row["final_answer"],
            ),
        })

        if (i + 1) % 10 == 0:
            print(
                f"Completed {i + 1} / "
                f"{len(experiment_df)}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\nAccuracy by condition:")

    summary = (
        results_df
        .groupby("condition")["correct"]
        .agg(["sum", "count", "mean"])
    )

    print(summary)

    pivot = results_df.pivot(
        index="question_id",
        columns="condition",
        values="correct",
    )

    test = cochrans_q(
        pivot[
            ["clean", "first", "middle", "last"]
        ].values
    )

    print("\nCochran's Q test")
    print("Q:", test.statistic)
    print("p-value:", test.pvalue)


if __name__ == "__main__":
    main()
