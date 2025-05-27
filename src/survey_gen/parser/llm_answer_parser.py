
from typing import List, Dict

from ..utilities.survey_classes.survey_objects import SurveyQuestion, QuestionAnswerTuple

from ..inference.survey_inference import batch_generation

from ..survey_manager import SurveyResult, LLMSurvey

from vllm import LLM

import pandas as pd

import yaml

from survey_gen.survey_manager import SurveyResult, LLMSurvey


DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."
DEFAULT_PROMPT: str = "Your task is to parse the correct answer option from an open text answer a LLM has given to survey questions. You will be provided with the possible answer options and the full text answer. Answer ONLY and EXACTLY with one of the possible answer options or 'INVALID', if the provided answer does give on of the options."

def json_parser_str(answer:str) -> Dict[str, str]:
    result_json = yaml.safe_load(answer)

    return result_json

def json_parse_all(survey_results: List[SurveyResult]) -> Dict[LLMSurvey, pd.DataFrame]:
    final_result = {}

    for survey_result in survey_results:
        answers = []
        for key, value in survey_result.results.items():
            #value:QuestionAnswerTuple
            parsed_answer = json_parser_str(value.answer)
            answers.append((key, value.question, *parsed_answer.values()))
        df = pd.DataFrame(answers, columns=["question_id", "question", *parsed_answer.keys()]).sort_values(by="question_id")
        final_result[survey_result.survey] = df

    return final_result

def json_parse_whole_survey_all(survey_results:List[SurveyResult], json_structure:List[str]) -> Dict[LLMSurvey, pd.DataFrame]:
    parsed_results =  json_parse_all(survey_results)
    
    all_results = {}

    for survey, df in parsed_results.items():
        long_df = pd.wide_to_long(df, 
                                stubnames=json_structure, 
                                i=["question_id"],
                                j='actual_question_id', 
                                sep='', 
                                suffix='\\d+').reset_index()
        
        num_rows_to_update = len(long_df)
        
        long_df.loc[0:num_rows_to_update, 'question_id'] = [survey_question.question_id for survey_question in survey._questions[0:num_rows_to_update]]
        long_df.loc[0:num_rows_to_update, 'question'] = [survey.generate_question_prompt(survey_question) for survey_question in survey._questions[0:num_rows_to_update]]
        long_df = long_df.drop(columns='question_id').rename(columns={'actual_question_id': 'question_id'})
        all_results[survey] = long_df
    
    return all_results


@staticmethod
def llm_parse_all(model:LLM, survey_results:List[SurveyResult], system_prompt:str = DEFAULT_SYSTEM_PROMPT, prompt:str = DEFAULT_PROMPT, use_structured_ouput:bool = False, batch_size:int = 2) -> Dict[LLMSurvey, pd.DataFrame]:
    #TODO LLM Parser in batches, same output as json parser
    results = []

    for survey_result in survey_results:
        answers = []
        for key, value in survey_result.results.items():
            pass