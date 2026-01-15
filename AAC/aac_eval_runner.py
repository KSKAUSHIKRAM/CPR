import csv
from View.evaluation_monitor import EvaluationMonitor

evaluator = EvaluationMonitor()

with open('Test/aac_english_test_cases.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        evaluator.record_normalization(row["raw_input"], row["normalized"], row["normalized"])
        evaluator.record_category(row["category"], row["category"])
        evaluator.record_pictogram([row["pictogram"]], [row["pictogram"]])
        evaluator.record_label([row["generated"]], row["generated"])
        evaluator.record_sentence(row["generated"], row["reference"])

evaluator.evaluate_all()
