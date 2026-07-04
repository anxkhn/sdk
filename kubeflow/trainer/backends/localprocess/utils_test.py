# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

import kubeflow.trainer.backends.localprocess.constants as local_constants
import kubeflow.trainer.backends.localprocess.utils as utils
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


def _sample_train_func(a, b):
    return a + b


def _no_args_train_func():
    return "done"


def _build_runtime() -> types.Runtime:
    runtime_trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="torch",
        image=local_constants.LOCAL_RUNTIME_IMAGE,
    )
    runtime_trainer.set_command(("python",))
    return types.Runtime(name="test-runtime", trainer=runtime_trainer)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="func_args are unpacked as keyword arguments",
            expected_status=SUCCESS,
            config={
                "func": _sample_train_func,
                "func_args": {"a": 1, "b": 2},
            },
            expected_output="_sample_train_func(**{'a': 1, 'b': 2})",
        ),
        TestCase(
            name="no func_args calls the function without arguments",
            expected_status=SUCCESS,
            config={
                "func": _no_args_train_func,
                "func_args": None,
            },
            expected_output="_no_args_train_func()",
        ),
        TestCase(
            name="raises when runtime has no trainer",
            expected_status=FAILED,
            config={
                "func": _sample_train_func,
                "func_args": None,
                "runtime": types.Runtime(name="no-trainer", trainer=None),
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="raises when train_func is not callable",
            expected_status=FAILED,
            config={
                "func": "not-callable",
                "func_args": None,
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_command_using_train_func(test_case: TestCase, tmp_path: Path):
    venv_dir = str(tmp_path)
    runtime = test_case.config.get("runtime", _build_runtime())
    train_job_name = "test-job"

    try:
        utils.get_command_using_train_func(
            runtime=runtime,
            train_func=test_case.config["func"],
            train_func_parameters=test_case.config["func_args"],
            venv_dir=venv_dir,
            train_job_name=train_job_name,
        )

        assert test_case.expected_status == SUCCESS

        func_file = tmp_path / local_constants.LOCAL_EXEC_FILENAME.format(train_job_name)
        generated_code = func_file.read_text()

        # The generated call must unpack func_args as keyword arguments (**),
        # matching the Kubernetes and Container backends. Passing the dict as a
        # single positional argument would break any function with named params.
        generated_call = generated_code.strip().splitlines()[-1]
        assert generated_call == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
