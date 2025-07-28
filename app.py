import streamlit as st
import pandas as pd
import os
import time
import google.generativeai as genai
import json

# === Configure Gemini API ===
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not GEMINI_API_KEY:
    st.error("Please add your Google Gemini API key to the Streamlit secrets.toml file or environment variables.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# === Streamlit Config ===
st.set_page_config(page_title="Quiz App for Grades 5 & 7", layout="centered")

# === Paths ===
BASE_PATH = "input"
GEN_PATH = "generated"

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
main_page = st.sidebar.radio("Go to", ["About", "Subjects", "Generate Extra Questions"], index=1)  # Default to "Subjects"

# === Utility Functions ===
def get_subjects():
    try:
        subjects = sorted([d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and d in ["Grade5", "Grade7"]])
        print(f"Found subjects: {subjects}")  # Debug
        if not subjects:
            st.error(f"No grade folders found in '{BASE_PATH}/'. Expected: input/Grade5, input/Grade7")
        return subjects
    except FileNotFoundError:
        st.error(f"Directory '{BASE_PATH}/' not found. Please create it and add Grade5 and Grade7 folders.")
        return []
    except Exception as e:
        st.error(f"Error accessing '{BASE_PATH}/': {str(e)}")
        return []

def get_subjects_in_grade(grade):
    folder = os.path.join(BASE_PATH, grade)
    print(f"Looking for subjects in: {folder}")  # Debug
    if not os.path.exists(folder):
        st.error(f"Directory '{folder}' does not exist. Please create it and add subject folders (e.g., Math).")
        return []
    try:
        subjects = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
        print(f"Found subjects in {grade}: {subjects}")  # Debug
        return subjects
    except Exception as e:
        st.error(f"Error accessing '{folder}': {str(e)}")
        return []

def get_lessons(grade, subject):
    folder = os.path.join(BASE_PATH, grade, subject)
    print(f"Looking for lessons in: {folder}")  # Debug
    if not os.path.exists(folder):
        st.error(f"Directory '{folder}' does not exist. Please create it and add CSV files.")
        return []
    try:
        lessons = sorted([f[:-4] for f in os.listdir(folder) if f.endswith(".csv")])
        print(f"Found lessons for {grade}/{subject}: {lessons}")  # Debug
        if not lessons:
            st.warning(f"No CSV files found in '{folder}'. Please add lesson CSV files (e.g., roots.csv).")
        return lessons
    except Exception as e:
        st.error(f"Error accessing '{folder}': {str(e)}")
        return []

def load_quiz(grade, subject, lesson):
    path = os.path.join(BASE_PATH, grade, subject, f"{lesson}.csv")
    print(f"Loading quiz from: {path}")  # Debug
    if not os.path.exists(path):
        st.error(f"Quiz file '{path}' not found.")
        return None
    try:
        df = pd.read_csv(path)
        required_columns = ["question", "option1", "option2", "option3", "option4", "answer"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV file '{path}' is missing required columns: {required_columns}")
            return None
        # Drop rows with any missing or invalid values in required columns
        df = df.dropna(subset=required_columns)
        df = df[df[required_columns].apply(lambda x: all(isinstance(val, str) and val.strip() for val in x), axis=1)]
        # Validate that 'answer' matches one of the options
        df = df[df.apply(lambda x: str(x["answer"]).strip() in [str(x[f"option{i}"]).strip() for i in range(1, 5)], axis=1)]
        if df.empty:
            st.error(f"No valid questions found in '{path}'. Ensure all rows have non-empty values for {required_columns} and 'answer' matches one of the options.")
            return None
        print(f"Loaded {len(df)} valid questions from {path}")  # Debug
        return df
    except Exception as e:
        st.error(f"Error reading '{path}': {str(e)}")
        return None

def load_generated_quiz(grade, subject, lesson):
    path = os.path.join(GEN_PATH, grade, subject, f"{lesson}_generated.csv")
    print(f"Loading generated quiz from: {path}")  # Debug
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        required_columns = ["question", "option1", "option2", "option3", "option4", "answer"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"Generated CSV file '{path}' is missing required columns: {required_columns}")
            return None
        df = df.dropna(subset=required_columns)
        df = df[df[required_columns].apply(lambda x: all(isinstance(val, str) and val.strip() for val in x), axis=1)]
        df = df[df.apply(lambda x: str(x["answer"]).strip() in [str(x[f"option{i}"]).strip() for i in range(1, 5)], axis=1)]
        if df.empty:
            st.error(f"No valid questions found in '{path}'. Ensure all rows have non-empty values for {required_columns} and 'answer' matches one of the options.")
            return None
        print(f"Loaded {len(df)} valid questions from {path}")  # Debug
        return df
    except Exception as e:
        st.error(f"Error reading '{path}': {str(e)}")
        return None

def save_generated_questions(grade, subject, lesson, questions):
    gen_dir = os.path.join(GEN_PATH, grade, subject)
    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, f"{lesson}_generated.csv")
    print(f"Saving generated questions to: {path}")  # Debug
    try:
        new_df = pd.DataFrame(questions)
        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_csv(path, index=False)
    except Exception as e:
        st.error(f"Error saving questions to '{path}': {str(e)}")

@st.cache_data(show_spinner=False)
def get_gemini_explanation(grade, subject, question, options, correct_answer, user_answer):
    prompt = f"""
You are a helpful tutor for {grade} students studying {subject}. Explain this multiple-choice question in a student-friendly way, appropriate for {grade} level.

Question: {question}
Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

The correct answer is: {correct_answer}
The student's answer was: {user_answer}

Explain why the correct answer is correct, and if the student was wrong, clarify the mistake in a way that helps them learn.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not fetch explanation: {str(e)}"

@st.cache_data(show_spinner=False)
def generate_questions(grade, subject, lesson, num_questions=5):
    prompt = f"""
You are an expert educator creating a quiz for {grade} students studying {subject}, specifically the topic {lesson}. Generate {num_questions} multiple-choice questions, each with 4 options and one correct answer. Ensure the questions are appropriate for {grade} level, clear, and educationally valuable. The 'answer' must exactly match one of the options. Return the output in JSON format with the following structure:

[
    {{
        "question": "Question text",
        "option1": "Option A",
        "option2": "Option B",
        "option3": "Option C",
        "option4": "Option D",
        "answer": "Option A"
    }},
    ...
]
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        questions = json.loads(text)
        return questions
    except Exception as e:
        st.error(f"⚠️ Failed to generate questions: {str(e)}")
        return []

def run_quiz(grade, subject, lesson, generated=False):
    df_full = load_generated_quiz(grade, subject, lesson) if generated else load_quiz(grade, subject, lesson)
    if df_full is None:
        st.warning("Quiz data not found or invalid. Please check the CSV file.")
        return

    max_qs = min(20, len(df_full))
    question_count = st.selectbox("How many questions do you want to attempt?", [5, 10, 15, 20], index=1)
    question_count = min(question_count, max_qs)

    df = df_full.sample(n=question_count, random_state=42).reset_index(drop=True)

    source = "Generated" if generated else "Original"
    st.subheader(f"{source} Quiz - {grade} - {subject} - {lesson.capitalize()}")

    quiz_key = f"{grade}_{subject}_{lesson}_{'gen' if generated else 'orig'}"
    # Reset session state if question count changes or data is invalid
    if quiz_key not in st.session_state or len(st.session_state[quiz_key]["answers"]) != len(df):
        st.session_state[quiz_key] = {
            "current_q": 0,
            "answers": [""] * len(df),
            "submitted": False,
            "show_explanation": [False] * len(df),
        }

    state = st.session_state[quiz_key]
    current_q = state["current_q"]

    if state["submitted"]:
        correct = [str(ans).strip().lower() for ans in df["answer"].tolist()]
        submitted = [str(a).strip().lower() for a in state["answers"]]
        score = sum([a == c for a, c in zip(submitted, correct)])
        st.success(f"✅ You scored {score} out of {len(df)}")

        for i in range(len(df)):
            q = df.iloc[i]
            user_ans = str(state["answers"][i] or "").strip()
            correct_ans = str(q["answer"]).strip()
            result = "✅ Correct" if user_ans.lower() == correct_ans.lower() else f"❌ Incorrect (Correct: {correct_ans})"

            st.markdown(f"---\n**Q{i+1}: {q['question']}**")
            st.markdown(f"Your answer: {user_ans}  \n{result}")

            # Button to toggle explanation
            if st.button(f"Explain Answer Q{i+1}", key=f"explain_{quiz_key}_{i}"):
                state["show_explanation"][i] = not state["show_explanation"][i]

            if state["show_explanation"][i]:
                with st.spinner(f"Getting explanation for Q{i+1}..."):
                    explanation = get_gemini_explanation(
                        grade, subject,
                        q["question"],
                        [q["option1"], q["option2"], q["option3"], q["option4"]],
                        correct_ans,
                        user_ans,
                    )
                    st.markdown(f"**Explanation:**\n{explanation}")
                time.sleep(1)

        st.button("Restart Quiz", key=f"restart_{quiz_key}", on_click=lambda: st.session_state.pop(quiz_key))
        return

    row = df.iloc[current_q]
    options = [str(row["option1"]), str(row["option2"]), str(row["option3"]), str(row["option4"])]
    # Debug: Log options and current answer
    print(f"Question {current_q + 1} options: {options}")
    print(f"Current answer in state: {state['answers'][current_q]}")
    # Validate options to prevent IndexError
    if not options or not all(opt and isinstance(opt, str) for opt in options):
        st.error(f"Invalid question data for Q{current_q + 1} in '{lesson}.csv'. Options: {options}")
        return
    # Validate answer
    if str(row["answer"]).strip() not in options:
        st.error(f"Invalid answer for Q{current_q + 1} in '{lesson}.csv'. Answer: {row['answer']} not in options: {options}")
        return
    question = f"Q{current_q + 1}: {row['question']}"

    try:
        # Ensure current answer is valid
        current_answer = state["answers"][current_q]
        if current_answer and current_answer not in options:
            print(f"Resetting invalid answer for Q{current_q + 1}: {current_answer} not in {options}")
            state["answers"][current_q] = ""
            current_answer = ""
        user_choice = st.radio(
            question,
            options,
            key=f"{quiz_key}_{current_q}",
            index=options.index(current_answer) if current_answer in options else 0,
        )
        state["answers"][current_q] = user_choice
    except ValueError as e:
        st.error(f"Error rendering question Q{current_q + 1}: {str(e)}. Options: {options}, Answer: {row['answer']}, Stored answer: {state['answers'][current_q]}")
        return

    col1, col2 = st.columns(2)
    with col1:
        if current_q < len(df) - 1:
            st.button("Next", key=f"next_{quiz_key}_{current_q}", on_click=lambda: advance_question(1, len(df), quiz_key))
        else:
            st.button("Submit Quiz", key=f"submit_{quiz_key}", on_click=lambda: submit_quiz(quiz_key))

    with col2:
        if current_q > 0:
            st.button("Previous", key=f"prev_{quiz_key}_{current_q}", on_click=lambda: advance_question(-1, len(df), quiz_key))

def advance_question(delta, total, quiz_key):
    state = st.session_state[quiz_key]
    new_q = state["current_q"] + delta
    if 0 <= new_q < total:
        state["current_q"] = new_q

def submit_quiz(quiz_key):
    st.session_state[quiz_key]["submitted"] = True

# === Main App Logic ===
if main_page == "About":
    st.subheader("📘 About This Quiz App")
    st.markdown("""
- This app is designed for Grade 5 and Grade 7 students.
- Choose your grade (Grade 5 or Grade 7), a subject (e.g., Math), and a lesson (e.g., Roots) to begin.
- Answer multiple-choice questions and get instant feedback.
- After submitting, view results and click "Explain Answer" for each question to see explanations powered by **Gemini AI**.
- You can also generate extra questions using AI to practice more.
""")

elif main_page == "Subjects":
    subjects = get_subjects()
    if not subjects:
        st.error("No grades found. Please create input/Grade5 and/or input/Grade7 folders.")
    else:
        grade = st.sidebar.selectbox("Choose Grade", subjects, index=subjects.index("Grade7") if "Grade7" in subjects else 0)
        grade_subjects = get_subjects_in_grade(grade)
        if grade_subjects:
            subject = st.sidebar.selectbox("Choose Subject", grade_subjects, index=grade_subjects.index("Math") if "Math" in grade_subjects else 0)
            lessons = get_lessons(grade, subject)
            if lessons:
                lesson = st.sidebar.selectbox("Choose Lesson", lessons)
                quiz_type = st.sidebar.radio("Quiz Type", ["Original", "Generated"])
                run_quiz(grade, subject, lesson, generated=(quiz_type == "Generated"))
            else:
                st.warning(f"No lessons (CSV files) found in 'input/{grade}/{subject}/'")
        else:
            st.warning(f"No subjects found in 'input/{grade}/'. Please add subject folders (e.g., Math).")

elif main_page == "Generate Extra Questions":
    st.subheader("Generate Extra Questions with Gemini AI")
    subjects = get_subjects()
    if not subjects:
        st.error("No grades found. Please create input/Grade5 and/or input/Grade7 folders.")
    else:
        grade = st.selectbox("Choose Grade", subjects, index=subjects.index("Grade7") if "Grade7" in subjects else 0)
        grade_subjects = get_subjects_in_grade(grade)
        if grade_subjects:
            subject = st.selectbox("Choose Subject", grade_subjects, index=grade_subjects.index("Math") if "Math" in grade_subjects else 0)
            lessons = get_lessons(grade, subject)
            if lessons:
                lesson = st.selectbox("Choose Lesson", lessons)
                num_questions = st.number_input("Number of Questions to Generate", min_value=1, max_value=10, value=5)
                if st.button("Generate Questions"):
                    with st.spinner("Generating questions with Gemini AI..."):
                        questions = generate_questions(grade, subject, lesson, num_questions)
                        if questions:
                            save_generated_questions(grade, subject, lesson, questions)
                            st.success(f"✅ {num_questions} questions generated and saved for {grade} - {subject} - {lesson}!")
                            st.markdown("You can now take the generated quiz by selecting 'Generated' under Quiz Type in the Subjects page.")
                        else:
                            st.error("Failed to generate questions. Please try again.")
            else:
                st.warning(f"No lessons (CSV files) found in 'input/{grade}/{subject}/'")
        else:
            st.warning(f"No subjects found in 'input/{grade}/'. Please add subject folders (e.g., Math).")