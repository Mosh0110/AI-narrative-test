"""Write one row per analysed participant x main question for the three tests (input to crossed_models.R).

Run from the repository root:  python3 mixed_models/export_long_data.py
Output: mixed_models/data/{gsm8k,folio,folio_control}_long.csv with columns
participant, question, model, problem, correct (0/1), rt (seconds; empty if no time was recorded).
`problem` is the GSM8K problem (sample_idx) or the FOLIO example (example_id), so a FOLIO problem
used in both the FOLIO test and the FOLIO control has the same identifier.
Participants are those flagged as included in data/participant_inclusion.csv (80 per test).
"""
import json
import os

import pandas as pd

LETTERS = {"A": 0, "B": 1, "C": 2, "D": 3}
OUT = "mixed_models/data"
inclusion = pd.read_csv("data/participant_inclusion.csv", keep_default_na=False)


def included(study):
    flags = inclusion[inclusion["study"] == study]
    return set(flags.loc[flags["included"] == 1, "record"])


def first(v):
    return v[0] if isinstance(v, list) and v else v


def gsm8k():
    questions = json.load(open("data/test_questions.json"))
    attention = {12, 22, 35}                                   # question numbers of the attention checks
    df = pd.read_excel("data/responses/Questionnaire Responses.xlsx", sheet_name=0)
    df = df.rename(columns={"Unnamed: 2": "participant", "Unnamed: 12": "answers", "Unnamed: 13": "final",
                            "Unnamed: 14": "times"})
    df = df[df["final"].astype(str).str.lower() == "yes"]
    keep = included("GSM8K")
    rows = []
    for _, r in df.iterrows():
        if "participant %s" % r["participant"] not in keep:
            continue
        answers, times = json.loads(r["answers"]), json.loads(r["times"])
        for qnum, q in enumerate(questions, start=1):
            if qnum in attention:
                continue
            answer = first(answers.get(str(qnum)))
            rows.append(dict(participant=r["participant"], question=qnum, model=q["model_name"], problem=q["sample_idx"],
                             correct=int(bool(answer) and LETTERS.get(str(answer).upper()) == int(q["achieved_html_pos"])),
                             rt=times.get(str(qnum))))
    return pd.DataFrame(rows)


def folio(study, responses_path, test_path):
    test = json.load(open(test_path))
    attention = {i + 1 for i in test["attention_check_indices"]}
    responses = pd.read_csv(responses_path)
    responses = responses[responses["is_complete"].str.lower() == "yes"]
    keep = included(study)
    rows = []
    for _, r in responses.iterrows():
        if "response_row_id %d" % r["response_row_id"] not in keep:
            continue
        answers, times = json.loads(r["answers_json"]), json.loads(r["time_per_question_s"])
        for qnum, q in enumerate(test["questions"], start=1):
            if qnum in attention:
                continue
            answer = first(answers.get(str(qnum)))
            rows.append(dict(participant=int(r["response_row_id"]), question=qnum, model=q["model_name"],
                             problem=q["example_id"],
                             correct=int(answer is not None and LETTERS.get(str(answer).upper()) == LETTERS[first(q["correct_answers"])]),
                             rt=times.get(str(qnum))))
    return pd.DataFrame(rows)


os.makedirs(OUT, exist_ok=True)
tests = {
    "gsm8k": gsm8k(),
    "folio": folio("FOLIO test", "data/folio/responses/folio_actual_test_responses.csv",
                   "data/folio/actual_test/folio_actual_test.json"),
    "folio_control": folio("FOLIO control", "data/folio/responses/folio_control_test_responses.csv",
                           "data/folio/control_test/folio_control_test.json"),
}
for name, d in tests.items():
    d.to_csv(os.path.join(OUT, name + "_long.csv"), index=False)
    print("%s: %d participants, %d questions, %d problems, %d responses, accuracy %.1f%%"
          % (name, d.participant.nunique(), d.question.nunique(), d.problem.nunique(), len(d), 100 * d.correct.mean()))
