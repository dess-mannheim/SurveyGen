from typing import List, Dict, Optional, Union, overload, Self, Tuple

from dataclasses import replace

from .utilities.survey_objects import AnswerOptions, QuestionnaireItem

from .utilities import constants, placeholder
from .utilities.constants import QuestionnaireType

from .utilities.utils import safe_format_with_regex

import pandas as pd

import random

import copy


from transformers import AutoTokenizer


class LLMQuestionnaire:
    """
    Main class for setting up and managing an LLM-based interview or survey.

    This class handles loading questions, preparing prompts, managing answer options,
    and generating prompt structures for different interview types.
    """

    DEFAULT_QUESTIONNAIRE_ID: str = "Questionnaire"

    DEFAULT_SYSTEM_PROMPT: str = (
        "You will be given questions and possible answer options for each. Please reason about each question before answering."
    )
    DEFAULT_TASK_INSTRUCTION: str = ""

    DEFAULT_JSON_STRUCTURE: List[str] = ["reasoning", "answer"]

    DEFAULT_PROMPT_STRUCTURE: str = (
        f"{placeholder.PROMPT_QUESTIONS}\n{placeholder.PROMPT_OPTIONS}"
    )

    def __init__(
        self,
        questionnaire_source=Union[str, pd.DataFrame],
        questionnaire_name: str = DEFAULT_QUESTIONNAIRE_ID,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        prompt: str = DEFAULT_PROMPT_STRUCTURE,
        verbose=False,
        seed: int = 42,
    ):
        """
        Initialize an LLMInterview instance. Either a path to a csv file or a pandas dataframe has to be provided

        Args:
            questionnaire_source (str): Path to the CSV file containing the interview structure and questions.
            interview_dataframe (pd.Dataframe): A pandas dataframe interview structure and questions.
            interview_name (str): Name/ID for the interview.
            system_prompt (str): System prompt to prepend to all questions.
            interview_instruction (str): Instructions that will be given to the model before asking the questions.
            verbose (bool): If True, enables verbose output.
            seed (int): Random seed for reproducibility.
        """
        random.seed(seed)

        if questionnaire_source is None:
            raise ValueError("Either a path or a dataframe have to be provided")

        self.load_questionnaire_format(questionnaire_source=questionnaire_source)

        self.verbose: bool = verbose

        self.questionnaire_name: str = questionnaire_name

        self.system_prompt: str = system_prompt
        self.prompt: str = prompt

        self._same_options = False

    def duplicate(self):
        """
        Create a deep copy of the current interview instance.

        Returns:
            LLMQuestionnaire: A deep copy of the current object.
        """
        return copy.deepcopy(self)

    def get_prompt_for_questionnaire_type(
        self,
        questionnaire_type: QuestionnaireType = QuestionnaireType.SINGLE_ITEM,
        item_id: int = 0,
        item_separator: str = "\n",
    ) -> Tuple[str, str]:
        """
        Generate the full prompt for a given interview type.

        Args:
            interview_type (InterviewType): The type of interview prompt to generate.

        Returns:
            str: The constructed prompt for the interview type.
        """
        options = ""
        automatic_output_instructions = ""
        if (
            questionnaire_type == QuestionnaireType.SINGLE_ITEM
            or questionnaire_type == QuestionnaireType.SEQUENTIAL
        ):
            question = self.generate_question_prompt(self._questions[item_id])

            if self._questions[item_id].answer_options:
                options = self._questions[item_id].answer_options.create_options_str()

                rgm = self._questions[item_id].answer_options.response_generation_method
                if rgm is None:  # by default, no response generation method is required
                    automatic_output_instructions = ""
                else:
                    automatic_output_instructions: str = rgm.get_automatic_prompt()
            else:
                options = ""
                automatic_output_instructions = ""

            format_dict = {
                placeholder.PROMPT_QUESTIONS: question,
                placeholder.PROMPT_OPTIONS: options,
                placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS: automatic_output_instructions,
            }

        elif questionnaire_type == QuestionnaireType.BATTERY:
            all_questions: List[str] = []
            for question in self._questions:
                current_question_prompt = self.generate_question_prompt(question)

                if question.answer_options:
                    options = question.answer_options.create_options_str()
                else:
                    options = ""
                format_dict = {
                    placeholder.PROMPT_OPTIONS: options,
                }
                current_question_prompt = safe_format_with_regex(
                    current_question_prompt, format_dict
                )
                all_questions.append(current_question_prompt)

            all_questions_str = item_separator.join(all_questions)
            if self._questions[item_id].answer_options:
                options = self._questions[item_id].answer_options.create_options_str()
                rgm = self._questions[item_id].answer_options.response_generation_method

                if rgm is None:  # by default, no response generation method is required
                    automatic_output_instructions = ""
                else:
                    automatic_output_instructions: str = rgm.get_automatic_prompt(
                        questions=self._questions
                    )
            else:
                options = ""
                automatic_output_instructions = ""

            format_dict = {
                placeholder.PROMPT_QUESTIONS: all_questions_str,
                placeholder.PROMPT_OPTIONS: options,
                placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS: automatic_output_instructions,
            }

        system_prompt = safe_format_with_regex(self.system_prompt, format_dict)
        prompt = safe_format_with_regex(self.prompt, format_dict)

        return system_prompt, prompt

    def calculate_input_token_estimate(
        self, model_id: str, questionnaire_type: QuestionnaireType = QuestionnaireType.SINGLE_ITEM
    ) -> int:
        """
        Estimate the number of input tokens for the prompt, given a model and interview type.
        Remember that the model also has to have enough context length to fit its own response
        in case of CONTEXT and ONE_PROMPT type.

        Args:
            model_id (str): Huggingface model id.
            interview_type (InterviewType): Type of interview prompt.

        Returns:
            int: Estimated number of input tokens.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        system_prompt, prompt = self.get_prompt_for_questionnaire_type(
            questionnaire_type=questionnaire_type
        )
        system_tokens = tokenizer.encode(system_prompt)
        tokens = tokenizer.encode(prompt)
        total_tokens = len(system_tokens) + len(tokens)

        return (
            total_tokens
            if questionnaire_type != QuestionnaireType.SEQUENTIAL
            else len(total_tokens) * 3
        )

    def get_questions(self) -> str:
        """
        Get the list of loaded interview questions.

        Returns:
            List[InterviewItem]: The loaded questions.
        """
        return self._questions

    def load_questionnaire_format(self, questionnaire_source: Union[str, pd.DataFrame]) -> Self:
        """
        Load interview questions from a CSV file.

        The CSV should have columns: interview_item_id, question_content
        Optionally it can also have question_stem.

        Args:
            interview_source (str or pd.Dataframe): Path to a valid CSV file or pd.Dataframe.

        Returns:
            Self: The updated instance with loaded questions.
        """
        questionnaire_questions: List[QuestionnaireItem] = []

        if questionnaire_source is None:
            raise ValueError("Either a path or a dataframe have to be provided")

        if type(questionnaire_source) == pd.DataFrame:
            df = questionnaire_source
        else:
            df = pd.read_csv(questionnaire_source)

        for _, row in df.iterrows():
            questionnaire_item_id = row[constants.QUESTIONNAIRE_ITEM_ID]
            # if constants.QUESTION in df.columns:
            #     question = row[constants.QUESTION]
            if constants.QUESTION_CONTENT in df.columns:
                questionnaire_question_content = row[constants.QUESTION_CONTENT]
            else:
                questionnaire_question_content = None

            if constants.QUESTION_STEM in df.columns:
                question_stem = row[constants.QUESTION_STEM]
            else:
                question_stem = None

            generated_questionnaire_question = QuestionnaireItem(
                item_id=questionnaire_item_id,
                question_content=questionnaire_question_content,
                question_stem=question_stem,
            )
            questionnaire_questions.append(generated_questionnaire_question)

        self._questions = questionnaire_questions
        return self

    # TODO Item order could be given by ids
    @overload
    def prepare_questionnaire(
        self,
        question_stem: Optional[str] = None,
        answer_options: Optional[AnswerOptions] = None,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self: ...

    @overload
    def prepare_questionnaire(
        self,
        question_stem: Optional[List[str]] = None,
        answer_options: Optional[Dict[int, AnswerOptions]] = None,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self: ...

    def prepare_questionnaire(
        self,
        question_stem: Optional[Union[str, List[str]]] = None,
        answer_options: Optional[Union[AnswerOptions, Dict[int, AnswerOptions]]] = None,
        prefilled_responses: Optional[Dict[int, str]] = None,
        randomized_item_order: bool = False,
    ) -> Self:
        """
        Prepare the interview by assigning question stems, answer options, and prefilled responses.

        Args:
            question_stem (str or List[str], optional): Single or list of question stems.
            answer_options (AnswerOptions or Dict[int, AnswerOptions], optional): Answer options for all or per question.
            global_options (bool): If True, the answer options will be specified once at the end of the task instructions. Otherwise, they will be specified once per question.
            prefilled_responses (Dict[int, str], optional): If you provide prefilled responses, they will be used
            to fill the answers instead of prompting the LLM for that question.
            randomized_item_order (bool): If True, randomize the order of questions.

        Returns:
            Self: The updated instance with prepared questions.
        """
        questionnaire_questions: List[QuestionnaireItem] = self._questions

        prompt_list = isinstance(question_stem, list)
        if prompt_list:
            assert len(question_stem) == len(
                questionnaire_questions
            ), "If a list of question stems is given, length of prompt and survey questions have to be the same"

        options_dict = False

        if isinstance(answer_options, AnswerOptions):
            self._same_options = True
            options_dict = False
        elif isinstance(answer_options, Dict):
            self._same_options = False
            options_dict = True

        updated_questions: List[QuestionnaireItem] = []

        if not prefilled_responses:
            prefilled_responses = {}
            # for survey_question in survey_questions:
            # prefilled_answers[survey_question.question_id] = None

        if not prompt_list and not options_dict:
            updated_questions = []
            for i in range(len(questionnaire_questions)):
                new_questionnaire_question = replace(
                    questionnaire_questions[i],
                    question_stem=(
                        question_stem
                        if question_stem
                        else questionnaire_questions[i].question_stem
                    ),
                    answer_options=answer_options,
                    prefilled_response=prefilled_responses.get(
                        questionnaire_questions[i].item_id
                    ),
                )
                updated_questions.append(new_questionnaire_question)

        elif not prompt_list and options_dict:
            for i in range(len(questionnaire_questions)):
                new_questionnaire_question = replace(
                    questionnaire_questions[i],
                    question_stem=(
                        question_stem
                        if question_stem
                        else questionnaire_questions[i].question_stem
                    ),
                    answer_options=answer_options.get(questionnaire_questions[i].item_id),
                    prefilled_response=prefilled_responses.get(
                        questionnaire_questions[i].item_id
                    ),
                )
                updated_questions.append(new_questionnaire_question)

        elif prompt_list and not options_dict:
            for i in range(len(questionnaire_questions)):
                new_questionnaire_question = replace(
                    questionnaire_questions[i],
                    question_stem=(
                        question_stem[i]
                        if question_stem
                        else questionnaire_questions[i].question_stem
                    ),
                    answer_options=answer_options,
                    prefilled_response=prefilled_responses.get(
                        questionnaire_questions[i].item_id
                    ),
                )
                updated_questions.append(new_questionnaire_question)
        elif prompt_list and options_dict:
            for i in range(len(questionnaire_questions)):
                new_questionnaire_question = replace(
                    questionnaire_questions[i],
                    question_stem=(
                        question_stem[i]
                        if question_stem
                        else questionnaire_questions[i].question_stem
                    ),
                    answer_options=answer_options.get(questionnaire_questions[i].item_id),
                    prefilled_response=prefilled_responses.get(
                        questionnaire_questions[i].item_id
                    ),
                )
                updated_questions.append(new_questionnaire_question)

        if randomized_item_order:
            random.shuffle(updated_questions)

        self._questions = updated_questions
        return self

    def generate_question_prompt(self, questionnaire_items: QuestionnaireItem) -> str:
        """
        Generate the prompt string for a single interview question.

        Args:
            interview_question (InterviewItem): The question to prompt.

        Returns:
            str: The formatted prompt for the question.
        """

        if questionnaire_items.question_stem:
            if placeholder.QUESTION_CONTENT in questionnaire_items.question_stem:
                format_dict = {
                    placeholder.QUESTION_CONTENT: questionnaire_items.question_content
                }
                question_prompt = safe_format_with_regex(
                    questionnaire_items.question_stem, format_dict
                )
            else:
                question_prompt = f"""{questionnaire_items.question_stem} {questionnaire_items.question_content}"""
        else:
            question_prompt = f"""{questionnaire_items.question_content}"""
        if questionnaire_items.answer_options:
            _options_str = questionnaire_items.answer_options.create_options_str()
            if _options_str is not None:
                safe_formatter = {placeholder.PROMPT_OPTIONS: _options_str}
                question_prompt = safe_format_with_regex(
                    question_prompt, safe_formatter
                )
        return question_prompt