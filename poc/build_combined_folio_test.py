import copy
import json
import random
import re
import statistics
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

import folio_parent_step_poc as fp


SEED = 42
BASE_QUESTIONS_PER_MODEL = 14
ATTENTION_CHECK_COUNT = 3
ATTENTION_CHECK_INDICES = [7, 15, 25]
ANSWER_LETTERS = "ABCD"
MAIN_BALANCED_COUNTS = (7, 7, 7, 7)
ATTENTION_TARGET_LETTERS = ("A", "B", "C")

QWEN_DIR = Path("poc/folio_qwen_validation_depth_ge6_first30")
R1_DIR = Path("poc/folio_r1_validation_depth_ge6_first30_dedicated")
OUTPUT_DIR = Path("poc/folio_combined_qwen_r1_depth_ge6_first30_q14_r14_attention3_seed42")
TEST_TITLE = "Step Dependency Questionnaire"
TEST_ID = "folio_actual_q14_r14_31q_attention3_seed42"


def load_json(path: Path):
    return json.loads(path.read_text())


def question_identity(question: dict) -> tuple:
    return (
        question["model_name"],
        question["story_key"],
        question["example_id"],
        question["sample_idx"],
        question["target_step_index"],
    )


def answer_letter(question: dict) -> str:
    return ANSWER_LETTERS[question["answer_indices"][0]]


def sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]
    return len(parts)


def rendered_question_has_logic_notation(formatted_question: dict) -> bool:
    patterns = [
        re.compile(r"[⊕∨∧→↔¬∀∃]"),
        re.compile(r"\\\\\(|\\\\\)|\\boxed|\\text|\\neg|\\land|\\lor|\\to|\\rightarrow|\\leftrightarrow|\\forall|\\exists"),
    ]
    fields = [
        formatted_question["target_step_text"],
        formatted_question.get("counterfactual_starting_text", ""),
        *[choice["text"] for choice in formatted_question["choices"]],
    ]
    return any(pattern.search(text or "") for pattern in patterns for text in fields)


def format_single_question(question: dict, merged_samples: dict) -> dict:
    return fp.format_test(
        {
            question["model_name"]: {
                str(question["sample_idx"]): merged_samples[question["model_name"]][str(question["sample_idx"])]
            }
        },
        [dict(question, question_number=1)],
    )["questions"][0]


def build_clean_main_pool(question_pool: list[dict], merged_samples: dict) -> list[dict]:
    clean_pool = []
    for original_index, question in enumerate(question_pool):
        formatted_question = format_single_question(question, merged_samples)
        if rendered_question_has_logic_notation(formatted_question):
            continue
        enriched_question = copy.deepcopy(question)
        enriched_question["_original_index"] = original_index
        enriched_question["_answer_letter"] = answer_letter(question)
        clean_pool.append(enriched_question)
    return clean_pool


def choose_best_subset_for_target_counts(question_pool: list[dict], target_counts: dict[str, int]) -> list[dict]:
    grouped = {letter: [] for letter in ANSWER_LETTERS}
    for question in question_pool:
        grouped[question["_answer_letter"]].append(question)

    for letter, target_count in target_counts.items():
        if len(grouped[letter]) < target_count:
            raise RuntimeError(f"Not enough questions for answer {letter}: need {target_count}, have {len(grouped[letter])}")

    best_subset = None
    best_score = None

    for chosen_a in combinations(grouped["A"], target_counts["A"]):
        for chosen_b in combinations(grouped["B"], target_counts["B"]):
            for chosen_c in combinations(grouped["C"], target_counts["C"]):
                for chosen_d in combinations(grouped["D"], target_counts["D"]):
                    subset = list(chosen_a) + list(chosen_b) + list(chosen_c) + list(chosen_d)
                    distinct_story_count = len({question["story_key"] for question in subset})
                    original_index_sum = sum(question["_original_index"] for question in subset)
                    max_original_index = max(question["_original_index"] for question in subset)
                    score = (distinct_story_count, -original_index_sum, -max_original_index)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_subset = subset

    best_subset.sort(key=lambda question: question["_original_index"])
    return [copy.deepcopy(question) for question in best_subset]


def choose_balanced_main_questions(
    qwen_pool: list[dict],
    r1_pool: list[dict],
) -> tuple[list[dict], list[dict]]:
    count_vectors = sorted(set(permutations((4, 4, 3, 3))))

    best_pair = None
    best_score = None
    for qwen_counts in count_vectors:
        r1_counts = tuple(MAIN_BALANCED_COUNTS[i] - qwen_counts[i] for i in range(4))
        if any(count < 0 for count in r1_counts):
            continue

        qwen_target = {ANSWER_LETTERS[i]: qwen_counts[i] for i in range(4)}
        r1_target = {ANSWER_LETTERS[i]: r1_counts[i] for i in range(4)}

        try:
            qwen_subset = choose_best_subset_for_target_counts(qwen_pool, qwen_target)
            r1_subset = choose_best_subset_for_target_counts(r1_pool, r1_target)
        except RuntimeError:
            continue

        qwen_story_count = len({question["story_key"] for question in qwen_subset})
        r1_story_count = len({question["story_key"] for question in r1_subset})
        total_index_sum = sum(question["_original_index"] for question in qwen_subset + r1_subset)
        score = (qwen_story_count + r1_story_count, -total_index_sum)

        if best_score is None or score > best_score:
            best_score = score
            best_pair = (qwen_subset, r1_subset)

    if best_pair is None:
        raise RuntimeError("Failed to find balanced main-question subsets")

    return best_pair


def choose_attention_questions(
    remaining_pool: list[dict],
    merged_samples: dict,
    used_story_keys: set[str],
) -> list[dict]:
    clean_by_letter = {letter: [] for letter in ANSWER_LETTERS}
    for question in remaining_pool:
        formatted_question = format_single_question(question, merged_samples)
        if rendered_question_has_logic_notation(formatted_question):
            continue
        enriched_question = copy.deepcopy(question)
        enriched_question["_answer_letter"] = answer_letter(question)
        clean_by_letter[enriched_question["_answer_letter"]].append(enriched_question)

    rng = random.Random(SEED)
    selected = []
    selected_ids = set()
    selected_story_keys = set()

    for letter in ATTENTION_TARGET_LETTERS:
        eligible = [
            question
            for question in clean_by_letter[letter]
            if question_identity(question) not in selected_ids
        ]
        if not eligible:
            raise RuntimeError(f"No clean attention-check candidates for answer {letter}")

        unseen_story_candidates = [
            question
            for question in eligible
            if question["story_key"] not in used_story_keys and question["story_key"] not in selected_story_keys
        ]
        candidate_pool = unseen_story_candidates or eligible
        chosen = copy.deepcopy(rng.choice(candidate_pool))
        selected.append(chosen)
        selected_ids.add(question_identity(chosen))
        selected_story_keys.add(chosen["story_key"])

    return selected


def build_interleaved_base(qwen_questions: list[dict], r1_questions: list[dict]) -> list[dict]:
    combined = []
    max_len = max(len(qwen_questions), len(r1_questions))
    for idx in range(max_len):
        if idx < len(qwen_questions):
            combined.append(copy.deepcopy(qwen_questions[idx]))
        if idx < len(r1_questions):
            combined.append(copy.deepcopy(r1_questions[idx]))
    return combined


def write_preview(path: Path, formatted_test: dict) -> None:
    lines = []
    for idx, question in enumerate(formatted_test["questions"]):
        lines.append(
            f"Q{question['question_number']} | sample={question['sample_idx']} | model={question['model_name']} | attention={question['is_attention_check']}"
        )
        lines.append(
            f"story_key={question['story_key']} | example_id={question['example_id']} | gold_label={question['gold_label']}"
        )
        lines.append(f"target_step_index={question['target_step_index']}")
        lines.append(f"target_step_text={question['target_step_text']}")
        lines.append(f"counterfactual_starting_text={question['counterfactual_starting_text']}")
        for choice in question["choices"]:
            lines.append(f"  {choice['choice_letter']}. step {choice['step_index']}: {choice['text']}")
        lines.append(f"correct={','.join(question['correct_answers'])}")
        lines.append("")
    path.write_text("\n".join(lines))


def build_stats(formatted_test: dict) -> dict:
    questions = formatted_test["questions"]
    model_counts = Counter(question["model_name"] for question in questions)
    answer_counts = Counter(question["correct_answers"][0] for question in questions)
    gold_counts = Counter(question["gold_label"] for question in questions)
    attention_model_counts = Counter(
        question["model_name"] for question in questions if question["is_attention_check"]
    )

    question_sample_keys = {
        (question["model_name"], question["sample_idx"], question["story_key"], question["example_id"])
        for question in questions
    }
    distinct_story_keys = sorted({question["story_key"] for question in questions})
    target_indices = [question["target_step_index"] for question in questions]
    target_sentence_counts = [sentence_count(question["target_step_text"]) for question in questions]
    choice_sentence_counts = [
        sentence_count(choice["text"])
        for question in questions
        for choice in question["choices"]
    ]
    target_char_counts = [len(question["target_step_text"]) for question in questions]
    choice_char_counts = [len(choice["text"]) for question in questions for choice in question["choices"]]

    source_step_counts = []
    for question in questions:
        sample = formatted_test["samples"][question["model_name"]][str(question["sample_idx"])]
        source_step_counts.append(len(sample["evaluation_steps"]) - 1)

    return {
        "settings": {
            "seed": SEED,
            "base_questions_per_model": BASE_QUESTIONS_PER_MODEL,
            "attention_check_count": ATTENTION_CHECK_COUNT,
            "attention_check_indices": ATTENTION_CHECK_INDICES,
        },
        "num_questions": len(questions),
        "num_main_questions": sum(not question["is_attention_check"] for question in questions),
        "num_attention_checks": sum(question["is_attention_check"] for question in questions),
        "num_models": len(model_counts),
        "num_distinct_question_samples": len(question_sample_keys),
        "num_distinct_story_keys": len(distinct_story_keys),
        "model_counts": dict(model_counts),
        "attention_check_model_counts": dict(attention_model_counts),
        "correct_answer_counts": dict(answer_counts),
        "gold_label_counts": dict(gold_counts),
        "target_step_index_stats": {
            "min": min(target_indices),
            "median": statistics.median(target_indices),
            "mean": statistics.mean(target_indices),
            "max": max(target_indices),
        },
        "source_trajectory_step_stats": {
            "min": min(source_step_counts),
            "median": statistics.median(source_step_counts),
            "mean": statistics.mean(source_step_counts),
            "max": max(source_step_counts),
        },
        "target_sentence_count_stats": {
            "min": min(target_sentence_counts),
            "median": statistics.median(target_sentence_counts),
            "mean": statistics.mean(target_sentence_counts),
            "max": max(target_sentence_counts),
        },
        "choice_sentence_count_stats": {
            "min": min(choice_sentence_counts),
            "median": statistics.median(choice_sentence_counts),
            "mean": statistics.mean(choice_sentence_counts),
            "max": max(choice_sentence_counts),
        },
        "target_char_count_stats": {
            "min": min(target_char_counts),
            "median": statistics.median(target_char_counts),
            "mean": statistics.mean(target_char_counts),
            "max": max(target_char_counts),
        },
        "choice_char_count_stats": {
            "min": min(choice_char_counts),
            "median": statistics.median(choice_char_counts),
            "mean": statistics.mean(choice_char_counts),
            "max": max(choice_char_counts),
        },
    }


def find_replacement_question(
    bad_question: dict,
    remaining_pool: list[dict],
    merged_samples: dict,
) -> dict:
    target_answer_letter = answer_letter(bad_question)
    candidates = []
    for candidate in remaining_pool:
        if candidate["model_name"] != bad_question["model_name"]:
            continue
        candidate_answer_letter = answer_letter(candidate)
        if candidate_answer_letter != target_answer_letter:
            continue

        formatted_candidate = format_single_question(candidate, merged_samples)
        if rendered_question_has_logic_notation(formatted_candidate):
            continue
        candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            f"No clean replacement found for model={bad_question['model_name']} answer={target_answer_letter}"
        )

    candidates.sort(key=lambda q: (q["sample_idx"], q["target_step_index"], q["story_key"], q["example_id"]))
    return copy.deepcopy(candidates[0])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    qwen_formatted = load_json(QWEN_DIR / "parent_step_detection_test.json")
    r1_formatted = load_json(R1_DIR / "parent_step_detection_test.json")
    qwen_source_questions = load_json(QWEN_DIR / "question_candidates.json")
    r1_source_questions = load_json(R1_DIR / "question_candidates.json")
    qwen_samples = load_json(QWEN_DIR / "sample_bank.json")
    r1_samples = load_json(R1_DIR / "sample_bank.json")

    merged_samples = copy.deepcopy(qwen_formatted["samples"])
    merged_samples.update(copy.deepcopy(r1_formatted["samples"]))

    qwen_clean_main_pool = build_clean_main_pool(qwen_source_questions, merged_samples)
    r1_clean_main_pool = build_clean_main_pool(r1_source_questions, merged_samples)
    qwen_base_questions, r1_base_questions = choose_balanced_main_questions(qwen_clean_main_pool, r1_clean_main_pool)

    old_max_final_questions = fp.MAX_FINAL_QUESTIONS
    try:
        fp.MAX_FINAL_QUESTIONS = None
        random.seed(SEED)
        qwen_full_pool = fp.generate_questions(qwen_samples)
        random.seed(SEED)
        r1_full_pool = fp.generate_questions(r1_samples)
    finally:
        fp.MAX_FINAL_QUESTIONS = old_max_final_questions

    base_ids = {
        question_identity(question)
        for question in (qwen_base_questions + r1_base_questions)
    }

    remaining_pool = {}
    for question in qwen_full_pool + r1_full_pool:
        qid = question_identity(question)
        if qid in base_ids:
            continue
        remaining_pool[qid] = copy.deepcopy(question)

    used_story_keys = {question["story_key"] for question in (qwen_base_questions + r1_base_questions)}
    extra_attention_questions = choose_attention_questions(list(remaining_pool.values()), merged_samples, used_story_keys)

    combined_questions = build_interleaved_base(qwen_base_questions, r1_base_questions)
    for insert_idx, question in zip(ATTENTION_CHECK_INDICES, extra_attention_questions):
        combined_questions.insert(insert_idx, copy.deepcopy(question))

    used_question_ids = {question_identity(question) for question in combined_questions}
    available_replacements = [
        copy.deepcopy(question)
        for qid, question in remaining_pool.items()
        if qid not in used_question_ids
    ]

    for idx, question in enumerate(combined_questions):
        formatted_candidate = format_single_question(question, merged_samples)
        if not rendered_question_has_logic_notation(formatted_candidate):
            continue

        replacement = find_replacement_question(question, available_replacements, merged_samples)
        replacement_id = question_identity(replacement)
        available_replacements = [candidate for candidate in available_replacements if question_identity(candidate) != replacement_id]
        combined_questions[idx] = replacement

    for question_number, question in enumerate(combined_questions, start=1):
        question["question_number"] = question_number

    formatted_test = fp.format_test(merged_samples, combined_questions)
    formatted_test["title"] = TEST_TITLE
    formatted_test["test_id"] = TEST_ID
    formatted_test["attention_check_indices"] = ATTENTION_CHECK_INDICES

    attention_question_numbers = {idx + 1 for idx in ATTENTION_CHECK_INDICES}
    for question in formatted_test["questions"]:
        question["is_attention_check"] = question["question_number"] in attention_question_numbers

    stats = build_stats(formatted_test)

    (OUTPUT_DIR / "parent_step_detection_test.json").write_text(json.dumps(formatted_test, indent=2))
    (OUTPUT_DIR / "formatted_test.json").write_text(json.dumps(formatted_test, indent=2))
    (OUTPUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    (OUTPUT_DIR / "test.html").write_text(
        fp.render_test_html_with_options(formatted_test, prolific_mode=True, show_alternative_step_box=False)
    )
    (OUTPUT_DIR / "test_non_prolific.html").write_text(
        fp.render_test_html_with_options(formatted_test, prolific_mode=False, show_alternative_step_box=False)
    )
    write_preview(OUTPUT_DIR / "question_preview.txt", formatted_test)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
