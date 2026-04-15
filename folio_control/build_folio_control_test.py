from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "poc"))

from folio_parent_step_poc import load_original_html_generator


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_control_test_curated"
DEFAULT_CONTROL_SUMMARY = None

CONTROL_MODEL_NAME = "control/anthropic/claude-opus-4.6"
DEFAULT_SHOW_ALTERNATIVE_STEP_BOX = False
SELECT_QUESTIONS_RANDOMLY = False
DEFAULT_TARGET_QUESTION_COUNT = None
DEFAULT_TARGET_AVERAGE_STEPS_PER_QUESTION = None

AVAILABLE_CONTROL_SAMPLES = [
    {
        "sample_idx": 0,
        "story_id": 183,
        "example_id": 527,
        "dag_path": Path(__file__).resolve().parent / "prover9_poc_output" / "story183_example527.control_dag.json",
        "steps_path": Path(__file__).resolve().parent / "claude_opus_outputs_v6" / "story183_example527.gpt54_content.txt",
    },
    {
        "sample_idx": 1,
        "story_id": 219,
        "example_id": 621,
        "dag_path": Path(__file__).resolve().parent / "manual_quality_check_fixed" / "story219_example621.control_dag.json",
        "steps_path": Path(__file__).resolve().parent / "manual_quality_check_fixed" / "claude_outputs" / "story219_example621.gpt54_content.txt",
    },
    {
        "sample_idx": 2,
        "story_id": 426,
        "example_id": 1210,
        "dag_path": Path(__file__).resolve().parent / "manual_quality_check_depth7_next2" / "story426_example1210.control_dag.json",
        "steps_path": Path(__file__).resolve().parent / "manual_quality_check_depth7_next2" / "claude_outputs" / "story426_example1210.gpt54_content.txt",
    },
]


def parse_claude_steps(path: Path) -> list[dict]:
    raw = path.read_text().strip()
    if path.suffix == ".json":
        obj = json.loads(raw)
        return obj["steps"]
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    obj = json.loads(raw)
    return obj["steps"]


def build_main_question(dag: dict) -> str:
    return "\n".join(
        [
            "Premises:",
            "",
            "\n\n".join(dag["premises"]),
            "",
            "Conclusion:",
            dag["conclusion"],
            "",
            "Determine whether the conclusion is entailed by the premises.",
        ]
    )


def build_control_sample_registry() -> dict[tuple[int, int], dict]:
    registry = {}
    for spec in AVAILABLE_CONTROL_SAMPLES:
        key = (spec["story_id"], spec["example_id"])
        registry[key] = spec
    return registry


def build_control_sample_registry_from_summary(summary_path: Path) -> dict[tuple[int, int], dict]:
    rows = json.loads(summary_path.read_text())
    registry = {}
    for row in rows:
        sample_output_dir = Path(row["output_dir"])
        sample_dir = Path(row["sample_dir"])
        spec = {
            "sample_idx": row["index"],
            "story_id": row["story_id"],
            "example_id": row["example_id"],
            "dag_path": sample_dir / "control_dag.json",
            "steps_path": sample_output_dir / f"story{row['story_id']}_example{row['example_id']}.steps.json",
        }
        registry[(row["story_id"], row["example_id"])] = spec
    return registry


def extract_sample_keys_from_actual_test_source(path: Path) -> list[tuple[int, int]]:
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and "questions" in obj:
        items = obj["questions"]
    elif isinstance(obj, dict) and "samples" in obj:
        items = []
        for model_samples in obj["samples"].values():
            if isinstance(model_samples, dict):
                items.extend(model_samples.values())
    else:
        raise ValueError(f"Unsupported actual test source format: {path}")

    sample_keys = []
    seen = set()
    for item in items:
        if "story_id" not in item or "example_id" not in item:
            continue
        key = (int(item["story_id"]), int(item["example_id"]))
        if key in seen:
            continue
        seen.add(key)
        sample_keys.append(key)

    if not sample_keys:
        raise ValueError(f"No story_id/example_id pairs found in {path}")
    return sample_keys


def resolve_sample_specs(actual_test_source: Path | None, control_summary: Path | None) -> list[dict]:
    if control_summary is None and actual_test_source is None:
        return AVAILABLE_CONTROL_SAMPLES

    registry = (
        build_control_sample_registry_from_summary(control_summary)
        if control_summary is not None
        else build_control_sample_registry()
    )

    if actual_test_source is None:
        return [registry[key] for key in sorted(registry.keys())]

    sample_specs = []
    for sample_key in extract_sample_keys_from_actual_test_source(actual_test_source):
        if sample_key not in registry:
            raise ValueError(f"Missing control artifacts for story_id={sample_key[0]} example_id={sample_key[1]}")
        sample_specs.append(registry[sample_key])
    return sample_specs


def load_control_samples(sample_specs: list[dict]) -> tuple[list[dict], dict]:
    samples = []
    samples_by_model = {CONTROL_MODEL_NAME: {}}
    for spec in sample_specs:
        dag = json.loads(spec["dag_path"].read_text())
        verbalized_steps = parse_claude_steps(spec["steps_path"])

        if len(verbalized_steps) != len(dag["verbalization_nodes"]):
            raise ValueError((spec["story_id"], spec["example_id"], len(verbalized_steps), len(dag["verbalization_nodes"])))

        solution_steps = [{"text": build_main_question(dag), "type": "root"}]
        step_num_to_node = {}
        direct_parent_steps = {}
        for step in verbalized_steps:
            step_num = step["step"]
            node = dag["verbalization_nodes"][step_num - 1]
            step_num_to_node[step_num] = node
            direct_parent_steps[step_num] = []
            solution_steps.append({"text": step["text"], "type": node["node_type"]})

        node_id_to_step_num = {node["node_id"]: step_num for step_num, node in step_num_to_node.items()}
        for step_num, node in step_num_to_node.items():
            direct_parent_steps[step_num] = [node_id_to_step_num[parent_id] for parent_id in node["direct_parent_node_ids"] if parent_id in node_id_to_step_num]

        links = []
        for target_step, parent_steps in direct_parent_steps.items():
            if target_step <= 1:
                continue
            for prior_step in range(1, target_step):
                links.append(
                    {
                        "omitted_step_index": prior_step,
                        "target_step_index": target_step,
                        "equivalent": "parent" if prior_step in parent_steps else "not_parent",
                    }
                )

        sample = {
            "sample_idx": spec["sample_idx"],
            "story_id": dag["story_id"],
            "example_id": dag["example_id"],
            "label": dag["label"],
            "manual_fix_notes": dag.get("manual_fix_notes", []),
            "main_question": build_main_question(dag),
            "model_name": CONTROL_MODEL_NAME,
            "solution_steps": solution_steps,
            "links": links,
            "direct_parent_steps": direct_parent_steps,
            "dag_path": str(spec["dag_path"]),
            "steps_path": str(spec["steps_path"]),
        }
        samples.append(sample)
        samples_by_model[CONTROL_MODEL_NAME][str(spec["sample_idx"])] = sample
    return samples, samples_by_model


def load_reference_choice_stats(path: Path | None) -> dict | None:
    if path is None:
        return None
    obj = json.loads(path.read_text())
    questions = obj["questions"]
    position_distance_sums = [0.0, 0.0, 0.0, 0.0]
    position_counts = [0, 0, 0, 0]
    correct_position_counts = [0, 0, 0, 0]
    for question in questions:
        target_step_index = question["target_step_index"]
        for pos, choice in enumerate(question["choices"]):
            position_distance_sums[pos] += target_step_index - choice["step_index"]
            position_counts[pos] += 1
        correct_position = ord(question["correct_answers"][0]) - 65
        correct_position_counts[correct_position] += 1
    return {
        "path": str(path),
        "num_questions": len(questions),
        "mean_distances_by_position": [
            position_distance_sums[pos] / position_counts[pos] for pos in range(4)
        ],
        "correct_position_counts": correct_position_counts,
        "attention_check_indices": obj.get("attention_check_indices", []),
    }


def scale_target_position_counts(reference_choice_stats: dict, target_question_count: int) -> list[int]:
    raw = [
        target_question_count * count / reference_choice_stats["num_questions"]
        for count in reference_choice_stats["correct_position_counts"]
    ]
    base = [int(x) for x in raw]
    remainder = target_question_count - sum(base)
    fractions = sorted(
        [(raw[i] - base[i], i) for i in range(4)],
        reverse=True,
    )
    for _, pos in fractions[:remainder]:
        base[pos] += 1
    return base


def generate_question_candidates(samples: list[dict], reference_choice_stats: dict | None) -> list[dict]:
    all_generated_candidates = []
    position_counts_for_generation_phase = [0, 0, 0, 0]

    for sample in samples:
        sample_idx = sample["sample_idx"]
        solution_steps = sample["solution_steps"]
        direct_parent_steps = sample["direct_parent_steps"]
        for target_step_index, parent_steps in sorted(direct_parent_steps.items()):
            eligible_parent_steps = [step_idx for step_idx in parent_steps if step_idx < target_step_index - 1]
            if len(eligible_parent_steps) == 0:
                continue
            prior_steps = list(range(1, target_step_index - 1))
            non_parent_steps = [step_idx for step_idx in prior_steps if step_idx not in eligible_parent_steps]
            if len(non_parent_steps) < 3:
                continue

            correct_choice_candidate = None
            selected_incorrect_choices = []

            possible_configurations = []
            for parent_step_index in eligible_parent_steps:
                parent_step_text = solution_steps[parent_step_index]["text"]
                for incorrect_choice_steps in itertools.combinations(non_parent_steps, 3):
                    candidate_step_indices = sorted([parent_step_index, *incorrect_choice_steps])
                    achieved_html_pos = candidate_step_indices.index(parent_step_index)
                    distance_profile = tuple(target_step_index - step_idx for step_idx in candidate_step_indices)
                    if reference_choice_stats is None:
                        score = distance_profile
                    else:
                        score = tuple(
                            abs(distance_profile[pos] - reference_choice_stats["mean_distances_by_position"][pos])
                            for pos in range(4)
                        )
                    possible_configurations.append(
                        {
                            "score": score,
                            "position_score": position_counts_for_generation_phase[achieved_html_pos],
                            "achieved_html_pos": achieved_html_pos,
                            "correct_choice_candidate": {
                                "step_index": parent_step_index,
                                "text": parent_step_text,
                                "is_parent": True,
                            },
                            "selected_incorrect_choices": list(incorrect_choice_steps),
                            "distance_profile": distance_profile,
                        }
                    )

            if possible_configurations:
                possible_configurations.sort(key=lambda x: (x["score"], x["position_score"], x["achieved_html_pos"]))
                best = possible_configurations[0]
                correct_choice_candidate = best["correct_choice_candidate"]
                selected_incorrect_choices = best["selected_incorrect_choices"]

            if correct_choice_candidate is None or len(selected_incorrect_choices) != 3:
                continue

            choices_data = [correct_choice_candidate] + [
                {
                    "step_index": step_idx,
                    "text": solution_steps[step_idx]["text"],
                    "is_parent": False,
                }
                for step_idx in selected_incorrect_choices
            ]
            choices_data.sort(key=lambda x: x["step_index"])
            achieved_html_pos = next(i for i, choice in enumerate(choices_data) if choice["is_parent"])
            position_counts_for_generation_phase[achieved_html_pos] += 1

            question_data = {
                "question_number": 0,
                "sample_idx": sample_idx,
                "model_name": sample["model_name"],
                "story_id": sample["story_id"],
                "example_id": sample["example_id"],
                "gold_label": sample["label"],
                "target_step_index": target_step_index,
                "target_step_text": solution_steps[target_step_index]["text"],
                "counterfactual_starting_text": "",
                "choices": choices_data,
                "answer_indices": [achieved_html_pos],
                "achieved_html_pos": achieved_html_pos,
                "dag_direct_parent_steps": parent_steps,
                "eligible_parent_steps": eligible_parent_steps,
                "num_steps_shown": target_step_index,
                "choice_distances": [target_step_index - choice["step_index"] for choice in choices_data],
            }
            all_generated_candidates.append(
                {
                    "question_data": question_data,
                    "achieved_html_pos": achieved_html_pos,
                }
            )
    return all_generated_candidates


def select_questions(
    all_generated_candidates: list[dict],
    target_question_count: int | None,
    target_average_steps_per_question: float | None,
    reference_choice_stats: dict | None,
) -> list[dict]:
    if not all_generated_candidates:
        raise ValueError("No valid control-question candidates generated.")

    if target_average_steps_per_question is not None and target_question_count is None:
        raise ValueError("target_average_steps_per_question requires target_question_count.")

    if target_question_count is None:
        selected = [candidate["question_data"] for candidate in all_generated_candidates]
    else:
        remaining_candidates = list(all_generated_candidates)
        final_selected = []
        final_html_position_counts = [0, 0, 0, 0]
        final_distance_sums = [0.0, 0.0, 0.0, 0.0]
        questions_selected_per_sample = defaultdict(int)
        running_step_sum = 0
        target_position_counts = (
            scale_target_position_counts(reference_choice_stats, target_question_count)
            if reference_choice_stats is not None
            else None
        )

        for _ in range(target_question_count):
            if not remaining_candidates:
                break
            scored_candidates = []
            for idx, candidate_info in enumerate(remaining_candidates):
                q_data = candidate_info["question_data"]
                achieved_pos = candidate_info["achieved_html_pos"]
                if target_position_counts is None:
                    score_html_overflow = final_html_position_counts[achieved_pos]
                    score_html_need = 0
                else:
                    projected_count = final_html_position_counts[achieved_pos] + 1
                    score_html_overflow = max(0, projected_count - target_position_counts[achieved_pos])
                    score_html_need = -(target_position_counts[achieved_pos] - final_html_position_counts[achieved_pos])
                score_sample = questions_selected_per_sample[q_data["sample_idx"]]
                if target_average_steps_per_question is None:
                    score_steps = 0
                else:
                    projected_avg = (running_step_sum + q_data["num_steps_shown"]) / (
                        len(final_selected) + 1
                    )
                    score_steps = abs(projected_avg - target_average_steps_per_question)
                if reference_choice_stats is None:
                    score_distances = 0
                else:
                    projected_distance_means = [
                        (final_distance_sums[pos] + q_data["choice_distances"][pos]) / (len(final_selected) + 1)
                        for pos in range(4)
                    ]
                    score_distances = sum(
                        abs(projected_distance_means[pos] - reference_choice_stats["mean_distances_by_position"][pos])
                        for pos in range(4)
                    )
                scored_candidates.append(
                    (
                        score_html_overflow,
                        score_html_need,
                        score_steps,
                        score_distances,
                        score_sample,
                        idx,
                        candidate_info,
                    )
                )

            scored_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
            _, _, _, _, _, original_idx, best_candidate = scored_candidates[0]
            q_data = best_candidate["question_data"]
            final_selected.append(q_data)
            final_html_position_counts[best_candidate["achieved_html_pos"]] += 1
            for pos in range(4):
                final_distance_sums[pos] += q_data["choice_distances"][pos]
            questions_selected_per_sample[q_data["sample_idx"]] += 1
            running_step_sum += q_data["num_steps_shown"]
            remaining_candidates.pop(original_idx)
        selected = final_selected

    selected.sort(key=lambda q: (q["sample_idx"], q["target_step_index"]))
    for i, q in enumerate(selected):
        q["question_number"] = i + 1
    return selected


def format_control_test(
    samples_by_model: dict,
    questions: list[dict],
    attention_check_indices: list[int],
    test_id: str,
) -> dict:
    instruction_text = (
        "You are presented with a problem and a step-by-step reasoning process. For each question, the reasoning is shown up to a target step (highlighted in yellow).\n\n"
        "<strong>Your task:</strong> From the four options provided, select the earlier step that directly supports the target step in the reasoning structure.\n\n"
        "Only one of the four options is a direct parent of the target step. The other three are earlier steps, but they are not direct parents of the target step."
    )

    test_format = {
        "title": "Step Dependency Questionnaire",
        "test_id": test_id,
        "instructions": instruction_text,
        "samples": samples_by_model,
        "questions": [],
        "attention_check_indices": attention_check_indices,
    }
    attention_question_numbers = {idx + 1 for idx in attention_check_indices}

    for q in questions:
        formatted_q = {
            "question_number": q["question_number"],
            "sample_idx": q["sample_idx"],
            "main_question": samples_by_model[q["model_name"]][str(q["sample_idx"])]["main_question"],
            "target_step_index": q["target_step_index"],
            "target_step_text": q["target_step_text"],
            "counterfactual_starting_text": "",
            "choices": [
                {
                    "choice_letter": chr(65 + i),
                    "step_index": c["step_index"],
                    "text": c["text"],
                }
                for i, c in enumerate(q["choices"])
            ],
            "correct_answers": [chr(65 + i) for i in q["answer_indices"]],
            "model_name": q["model_name"],
            "story_id": q["story_id"],
            "example_id": q["example_id"],
            "gold_label": q["gold_label"],
            "dag_direct_parent_steps": q["dag_direct_parent_steps"],
            "num_steps_shown": q["num_steps_shown"],
            "is_attention_check": q["question_number"] in attention_question_numbers,
        }
        test_format["questions"].append(formatted_q)
    return test_format


def render_control_test_html(
    formatted_test: dict,
    show_alternative_step_box: bool,
    prolific_mode: bool = True,
) -> str:
    generate_html_test, namespace = load_original_html_generator()
    namespace["samples_by_model"] = {
        model_name: {int(sample_idx): sample for sample_idx, sample in samples.items()}
        for model_name, samples in formatted_test["samples"].items()
    }
    namespace["attention_check_indices"] = formatted_test.get("attention_check_indices", [])
    html = generate_html_test(formatted_test, show_alternative_step_box, prolific_mode=prolific_mode)
    submission_title_line = "                testTitle: document.title,\n"
    assert submission_title_line in html
    html = html.replace(
        submission_title_line,
        "                testTitle: "
        f"{json.dumps(formatted_test['title'])},\n"
        "                testId: "
        f"{json.dumps(formatted_test['test_id'])},\n",
        1,
    )
    html = html.replace(
        ".main-problem { background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
        ".main-problem { background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }\n"
        "        .main-problem p { white-space: pre-line; }",
        1,
    )
    if "<meta charset=\"UTF-8\">" not in html and "<meta charset=\"utf-8\">" not in html:
        html = html.replace("<head>", "<head>\n    <meta charset=\"UTF-8\">", 1)
    return html


def write_preview(formatted_test: dict, output_path: Path) -> None:
    lines = []
    for question in formatted_test["questions"]:
        lines.append(
            f"Q{question['question_number']} | sample={question['sample_idx']} | story_id={question['story_id']} | example_id={question['example_id']}"
        )
        lines.append(f"target_step_index={question['target_step_index']} | num_steps_shown={question['num_steps_shown']}")
        lines.append(f"target_step_text={question['target_step_text']}")
        lines.append(f"dag_direct_parent_steps={question['dag_direct_parent_steps']}")
        for choice in question["choices"]:
            lines.append(f"  {choice['choice_letter']}. step {choice['step_index']}: {choice['text']}")
        lines.append(f"correct={','.join(question['correct_answers'])}")
        lines.append("")
    output_path.write_text("\n".join(lines))


def build_summary(
    samples: list[dict],
    questions: list[dict],
    all_generated_candidates: list[dict],
    actual_test_source: str | None,
    show_alternative_step_box: bool,
    target_question_count: int | None,
    target_average_steps_per_question: float | None,
    reference_choice_stats: dict | None,
) -> dict:
    per_sample = []
    for sample in samples:
        target_counts = []
        for target_step_index, parent_steps in sorted(sample["direct_parent_steps"].items()):
            eligible_parent_steps = [step_idx for step_idx in parent_steps if step_idx < target_step_index - 1]
            prior_steps = list(range(1, target_step_index - 1))
            non_parent_steps = [step_idx for step_idx in prior_steps if step_idx not in eligible_parent_steps]
            target_counts.append(
                {
                    "target_step_index": target_step_index,
                    "parent_steps": parent_steps,
                    "eligible_parent_steps": eligible_parent_steps,
                    "num_parent_steps": len(parent_steps),
                    "num_eligible_parent_steps": len(eligible_parent_steps),
                    "num_non_parent_prior_steps": len(non_parent_steps),
                    "is_viable": len(eligible_parent_steps) >= 1 and len(non_parent_steps) >= 3,
                }
            )
        per_sample.append(
            {
                "sample_idx": sample["sample_idx"],
                "story_id": sample["story_id"],
                "example_id": sample["example_id"],
                "label": sample["label"],
                "manual_fix_notes": sample["manual_fix_notes"],
                "num_solution_steps": len(sample["solution_steps"]) - 1,
                "num_viable_targets": sum(item["is_viable"] for item in target_counts),
                "target_details": target_counts,
            }
        )

    avg_steps = 0 if not questions else sum(q["num_steps_shown"] for q in questions) / len(questions)
    return {
        "settings": {
            "actual_test_source": actual_test_source,
            "reference_choice_stats_source": None if reference_choice_stats is None else reference_choice_stats["path"],
            "show_alternative_step_box": show_alternative_step_box,
            "select_questions_randomly": SELECT_QUESTIONS_RANDOMLY,
            "target_question_count": target_question_count,
            "target_average_steps_per_question": target_average_steps_per_question,
        },
        "num_samples": len(samples),
        "num_generated_candidates": len(all_generated_candidates),
        "num_questions": len(questions),
        "average_steps_per_question": avg_steps,
        "per_sample": per_sample,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--actual_test_source", type=Path, default=None)
    parser.add_argument("--control_summary", type=Path, default=DEFAULT_CONTROL_SUMMARY)
    parser.add_argument("--reference_test_source", type=Path, default=None)
    parser.add_argument(
        "--show_alternative_step_box",
        action="store_true",
        default=DEFAULT_SHOW_ALTERNATIVE_STEP_BOX,
    )
    parser.add_argument("--target_question_count", type=int, default=DEFAULT_TARGET_QUESTION_COUNT)
    parser.add_argument(
        "--target_average_steps_per_question",
        type=float,
        default=DEFAULT_TARGET_AVERAGE_STEPS_PER_QUESTION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_specs = resolve_sample_specs(args.actual_test_source, args.control_summary)
    samples, samples_by_model = load_control_samples(sample_specs)
    reference_choice_stats = load_reference_choice_stats(args.reference_test_source)
    all_generated_candidates = generate_question_candidates(samples, reference_choice_stats)
    questions = select_questions(
        all_generated_candidates,
        target_question_count=args.target_question_count,
        target_average_steps_per_question=args.target_average_steps_per_question,
        reference_choice_stats=reference_choice_stats,
    )
    attention_check_indices = [] if reference_choice_stats is None else reference_choice_stats["attention_check_indices"]
    formatted_test = format_control_test(samples_by_model, questions, attention_check_indices, output_dir.name)
    summary = build_summary(
        samples,
        questions,
        all_generated_candidates,
        actual_test_source=None if args.actual_test_source is None else str(args.actual_test_source),
        show_alternative_step_box=args.show_alternative_step_box,
        target_question_count=args.target_question_count,
        target_average_steps_per_question=args.target_average_steps_per_question,
        reference_choice_stats=reference_choice_stats,
    )

    (output_dir / "sample_bank.json").write_text(json.dumps(samples, indent=2))
    (output_dir / "samples_by_model.json").write_text(json.dumps(samples_by_model, indent=2))
    (output_dir / "test_questions.json").write_text(json.dumps(questions, indent=2))
    (output_dir / "parent_step_detection_test.json").write_text(json.dumps(formatted_test, indent=2))
    (output_dir / "formatted_test.json").write_text(json.dumps(formatted_test, indent=2))
    (output_dir / "test.html").write_text(
        render_control_test_html(
            formatted_test,
            show_alternative_step_box=args.show_alternative_step_box,
            prolific_mode=True,
        )
    )
    (output_dir / "test_non_prolific.html").write_text(
        render_control_test_html(
            formatted_test,
            show_alternative_step_box=args.show_alternative_step_box,
            prolific_mode=False,
        )
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_preview(formatted_test, output_dir / "question_preview.txt")

    print(json.dumps(summary, indent=2))
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
