import concurrent.futures
import os
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

from deepeval import evaluate
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from tests.custom_judge import CustomJudge, CustomJudge2


def evaluate_wrapper(name):
    print(f"Starting {name}\n")
    json_str = "{\"score\": 10,\"reason\": \"Reason.\"}"
    judge = CustomJudge2("test", json_str)
    metric = GEval(
        name="Test Metric",
        evaluation_steps=["Step"],
        model=judge,
        evaluation_params=[LLMTestCaseParams.INPUT]
    )
    if name == "Run 1":
        test_case = LLMTestCase(
            input="Who is the current president of the United States of America?",
            actual_output="Joe Biden",
        )
    else:
        test_case = LLMTestCase(
            input="Why did the chicken cross the road?",
            # Replace this with your actual LLM application
            actual_output="Quite frankly, I don't want to know..."
        )
    dataset = EvaluationDataset(test_cases=[test_case])
    print(f"{name} evaluate call\n")

    evaluate(test_cases=dataset.test_cases, metrics=[metric],
             print_results=True,
             verbose_mode=True,
             ignore_errors=False)  # Call the black box function
    print(f"{name} evaluate finished\n")

@patch('deepeval.test_run.test_run.os.getenv', autospec=True)
def test_evaluate_can_execute_in_parallel(mock_getenv):
    def side_effect_function(param, default=None):
        return_values = {
            ("DEEPEVAL_RESULTS_FOLDER", None): Path(__file__).resolve().parents[0] / "data",
            (PYTEST_RUN_TEST_NAME, f"test_case_0"): f"test_case_0",
            ("DEEPEVAL_UNIQUE_ID", None): str(uuid.uuid4()),
            ("DEEPEVAL_TELEMETRY_OPT_OUT", None): "YES",
            ("ERROR_REPORTING", None): "NO"
        }
        return return_values.get((param, default), 'default_result')

    mock_getenv.side_effect = side_effect_function


    # evaluate_wrapper("Run 1")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            # Schedule the first evaluate run
            future1 = executor.submit(evaluate_wrapper, "Run 1")

            # Schedule the second evaluate run
            future2 = executor.submit(evaluate_wrapper, "Run 2")
        except Exception as e:
            print(e)

        # future1.result()
        # future2.result()
        # Wait for both futures to complete
        concurrent.futures.wait([future1, future2], return_when=concurrent.futures.ALL_COMPLETED)

