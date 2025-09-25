import streamlit as st
from surveygen.llm_interview import LLMInterview
from surveygen.survey_manager import conduct_survey_question_by_question, conduct_survey_in_context, conduct_whole_survey_one_prompt, SurveyOptionGenerator, SurveyCreator
from surveygen.utilities.constants import InterviewType
from gui_elements.paginator import paginator
import time


#CONSTANTS FOR FIELDS
system_prompt_field = "System prompt"
interview_instruction_field = "Interview instructions"

st.set_page_config(layout="wide")
st.title("Generate Prompt")
st.write(
    "This interface allows you to inspect and change the system prompt and primary instructions for the model."
)
st.divider()

#FOR DEBUGGING
# if "interviews" not in st.session_state:
#     st.session_state.interviews = [SurveyCreator().from_path(survey_path="/home/maxi/Documents/SurveyGen/surveys/ANES.csv", questionnaire_path="/home/maxi/Documents/SurveyGen/surveys/ANES_PERSONAS.csv")]
if "interviews" not in st.session_state:
    st.error("You need to first upload a questionnaire and the population you want to survey.")
    st.stop()

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

text_field_ids = [system_prompt_field, interview_instruction_field]

current_interview_id = paginator(st.session_state.interviews, "current_interview_index_prompt")

st.divider()

def process_text_inputs(text_input: str, field_id: str) -> str:
    
    if field_id == system_prompt_field:
        st.session_state.interviews[current_interview_id].system_prompt = text_input
    if field_id == interview_instruction_field:
        st.session_state.interviews[current_interview_id].interview_instruction = text_input

def handle_text_change(field_id: str):
    """
    This single callback handles changes from any text field.
    It reads the input from session state using the unique key,
    processes it, and saves the output to session state.
    """
    input_key = f"input_{field_id}"
    
    with st.spinner(f"Processing {field_id}..."):
        #time.sleep(0.5) # Simulate work
        process_text_inputs(st.session_state[input_key], field_id)


if "interviews" in st.session_state and st.session_state.interviews is not None:
    try:
        interview = st.session_state.interviews[current_interview_id]
    except IndexError:
        st.error("Index is out of range. Resetting to the first item.")
        current_interview_id = 0
        interview = st.session_state.interviews[current_interview_id]

    current_prompt = interview.get_prompt_for_interview_type(InterviewType.CONTEXT)

    col_options, col_prompt_display = st.columns(2)

    with col_options:
        st.subheader("⚙️ Configuration")

    # We loop  dynamically create the UI elements
    for field_id in text_field_ids:
        input_key = f"input_{field_id}"

        if field_id == system_prompt_field:
            st.session_state[input_key] = interview.system_prompt
        if field_id == interview_instruction_field:
            st.session_state[input_key] = interview.interview_instruction
        
        with col_options:
            #st.subheader("⚙️ Configuration")
            st.text_area(
                label=field_id,
                key=input_key,
                on_change=handle_text_change,
                kwargs={'field_id': field_id} 
            )
        
    # Place the corresponding output in the second column
    with col_prompt_display:
        st.subheader("📄 Live Preview")

    # --- The Dynamic Preview Logic ---
    # This block re-runs on every widget interaction.
        with st.container(border=True):
            current_prompt = interview.get_prompt_for_interview_type(InterviewType.CONTEXT)
            current_prompt = current_prompt.replace("\n", "  \n")
            st.write(current_prompt)
    if st.button("Confirm base structure", type="primary", use_container_width=True):
        pass
else:
    st.warning("No data found. Please upload a CSV file on the 'Upload CSV' page first.")


