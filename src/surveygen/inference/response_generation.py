import warnings
from abc import ABC
from typing import List, Dict, Optional

from ..utilities import prompt_templates, constants


# --- Answer Production Base Classes ---
# TODO: make the answer production methods compatible with being passed as a list


class ResponseGenerationMethod(ABC):
    """Abstract base class for constraining the model output, e.g., for closed-ended survey questions."""

    # NOTE that validation is not required anymore, since we rely on inheritance instead


class JSONResponseGenerationMethod(ResponseGenerationMethod):
    def __init__(
        self,
        json_fields: List[str] | Dict[str, str],  # required
        constraints: Optional[Dict[str, List[str]]] = None,  # remains optional
        automatic_system_prompt: bool = False,
        system_prompt_template: str = prompt_templates.SYSTEM_JSON_DEFAULT,
        output_index_only: bool = False,
    ):
        """
        Base class for constraining the model output using JSON Schema

        Attributes:
            json_fields: List of field names for JSON output, optionally as dicts of format {"field_name": "explanation"}
            constraints: Optional constraints for field values
            automatic_system_prompt: If a instruction to only output in the required json format should be added to the system prompt
            system_prompt_template: Template to use for formatting the system prompt, e.g., from `..utilities.prompt_templates`
            output_index_only: If True, constrain output to answer option index rather then the full text of each answer option
        """
        super().__init__()
        if constraints is not None:
            if isinstance(json_fields, dict):
                difference = set(constraints.keys()) - set(json_fields.keys())
            else:
                difference = set(constraints.keys()) - set(json_fields)
            if len(difference) > 0:
                warnings.warn(
                    f"Constraints specified for non-existing fields: {difference}.",
                    RuntimeWarning,
                )
        self.json_fields = json_fields
        self.constraints = constraints
        self.automatic_system_prompt = automatic_system_prompt
        self.system_prompt_template = system_prompt_template
        self.output_index_only = output_index_only


class ChoiceResponseGenerationMethod(ResponseGenerationMethod):
    def __init__(
        self,
        allowed_choices: List[str],  # required
        automatic_system_prompt: bool = False,
        system_prompt_template: str = prompt_templates.LIST_OPTIONS_DEFAULT,
        output_index_only: bool = False,
    ):
        """
        Base class for constraining the model output using a Choice between answer options

        Attributes:
            allowed_choices: List of allowed choices for choice output
            automatic_system_prompt: If a instruction to only output in the required json format should be added to the system prompt
            system_prompt_template: Template to use for formatting the system prompt, e.g., from `..utilities.prompt_templates`
            output_index_only: If True, constrain output to answer option index rather then the full text of each answer option
        """
        super().__init__()
        self.allowed_choices = allowed_choices
        self.automatic_system_prompt = automatic_system_prompt
        self.system_prompt_template = system_prompt_template
        self.output_index_only = output_index_only  # TODO: implement


class LogprobResponseGenerationMethod(ResponseGenerationMethod):
    def __init__(
        self,
        token_position: int = 0,
        token_limit: int = 1,
        top_logprobs: int = 20,  # the OpenAI API default, local vllm deployments might give you more
        allowed_choices: Optional[List[str]] = None,
        ignore_reasoning: bool = True,
        automatic_system_prompt: bool = False,
        system_prompt_template: str = prompt_templates.SYSTEM_SINGLE_ANSWER,
        output_index_only: bool = False,
    ):
        """
        Base class for constraining the model output by requesting token proabilities

        Attributes:
            token_position: At which position in the output to capture the logprobs, use `0` for first-token probabilities (default)
            token_limit: Overwrite the number of output tokens, e.g., only produce a single token for first-token probabilities (default)
            top_logprobs: How many of the logprobs to consider, OpenAI supports at most 20
            allowed_choices: If not None, restrict output additionally with `guided_choice`
            ignore_reasoning: If True, only consider tokens after the reasoning output, i.e., after </think>
            automatic_system_prompt: If a instruction to only output in the required json format should be added to the system prompt
            system_prompt_template: Template to use for formatting the system prompt, e.g., from `..utilities.prompt_templates`
            output_index_only: If True, constrain output to answer option index rather then the full text of each answer option
        """
        super().__init__()
        self.token_position = token_position
        self.token_limit = token_limit
        self.top_logprobs = top_logprobs
        self.allowed_choices = allowed_choices  # same name enables re-using code from Choice_AnswerProductionMethod
        self.ignore_reasoning = ignore_reasoning
        self.automatic_system_prompt = automatic_system_prompt
        self.system_prompt_template = system_prompt_template
        self.output_index_only = output_index_only  # TODO: implement


# --- Specific Answer Production Methods ---


class JSONSingleResponseGenerationMethod(JSONResponseGenerationMethod):
    def __init__(
        self,
        automatic_system_prompt: bool = False,
        output_index_only: bool = False,
    ):
        """Answer Production Method: Structured Outputs"""

        super().__init__(
            json_fields={"answer": constants.OPTIONS_ADJUST},
            constraints={"answer": constants.OPTIONS_ADJUST},
            automatic_system_prompt=automatic_system_prompt,
            system_prompt_template=prompt_templates.SYSTEM_JSON_SINGLE_ANSWER,
            output_index_only=output_index_only,
        )


class JSONReasoningResponseGenerationMethod(JSONResponseGenerationMethod):
    def __init__(
        self,
        automatic_system_prompt: bool = False,
        output_index_only: bool = False,
    ):
        """Answer Production Method: Structured Outputs with Reasoning"""

        json_fields = {
            "reasoning": "your reasoning about the answer options",
            "answer": constants.OPTIONS_ADJUST,
        }

        super().__init__(
            json_fields=json_fields,
            constraints={"answer": constants.OPTIONS_ADJUST},
            automatic_system_prompt=automatic_system_prompt,
            system_prompt_template=prompt_templates.SYSTEM_JSON_REASONING,
            output_index_only=output_index_only,
        )


class JSONAllOptionsResponseGenerationMethod(JSONResponseGenerationMethod):
    def __init__(
        self,
        automatic_system_prompt: bool = False,
        output_index_only: bool = False,
    ):
        """Answer Production Method: Structured Outputs All Options"""

        super().__init__(
            #will be set when given to answer options
            json_fields=None,
            constraints=None,
            #Variables
            automatic_system_prompt=automatic_system_prompt,
            system_prompt_template=prompt_templates.SYSTEM_JSON_ALL_OPTIONS,
            output_index_only=output_index_only,
        )
