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


# === Login Function ===
def check_login():
    valid_users = ["smiley", "Smiley", "SMILEY", "Pandu", "pandu", "PANDU", "haniya", "Haniya", "HANIYA", "HAYAN",
                   "Hayan", "hayan"]
    valid_password = "1234"

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["login_error"] = ""

    if not st.session_state["logged_in"]:
        st.title("Login to Quiz App")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Login"):
            if username in valid_users and password == valid_password:
                st.session_state["logged_in"] = True
                st.session_state["login_error"] = ""
                st.rerun()  # Rerun to show main app
            else:
                st.session_state["login_error"] = "Invalid username or password. Please try again."
        if st.session_state["login_error"]:
            st.error(st.session_state["login_error"])
        return False
    return True


# === Utility Functions ===
def get_subjects():
    try:
        subjects = sorted([d for d in os.listdir(BASE_PATH) if
                           os.path.isdir(os.path.join(BASE_PATH, d)) and d in ["Grade5", "Grade7"]])
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
        df = df[
            df.apply(lambda x: str(x["answer"]).strip() in [str(x[f"option{i}"]).strip() for i in range(1, 5)], axis=1)]
        if df.empty:
            st.error(
                f"No valid questions found in '{path}'. Ensure all rows have non-empty values for {required_columns} and 'answer' matches one of the options.")
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
        st.warning(f"Generated quiz file '{path}' not found. Please generate questions first.")
        return None
    try:
        df = pd.read_csv(path)
        required_columns = ["question", "option1", "option2", "option3", "option4", "answer"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"Generated CSV file '{path}' is missing required columns: {required_columns}")
            return None
        # Drop rows with any missing or invalid values in required columns
        df = df.dropna(subset=required_columns)
        df = df[df[required_columns].apply(lambda x: all(isinstance(val, str) and val.strip() for val in x), axis=1)]
        # Validate that 'answer' matches one of the options
        invalid_rows = []
        for idx, row in df.iterrows():
            answer = str(row["answer"]).strip()
            options = [str(row[f"option{i}"]).strip() for i in range(1, 5)]
            if answer not in options:
                invalid_rows.append((idx, row["question"], answer, options))
        if invalid_rows:
            print(f"Invalid rows in '{path}': {invalid_rows}")  # Debug
            df = df[df.apply(lambda x: str(x["answer"]).strip() in [str(x[f"option{i}"]).strip() for i in range(1, 5)],
                             axis=1)]
        if df.empty:
            st.error(
                f"No valid questions found in '{path}'. Ensure all rows have non-empty values for {required_columns} and 'answer' matches one of the options.")
            return None
        print(f"Loaded {len(df)} valid questions from {path}:")
        print(df.to_dict(orient='records'))  # Debug: Log loaded questions
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
        # Ensure all questions have required columns, add empty explanation if missing
        for q in questions:
            if "explanation" not in q:
                q["explanation"] = ""
        new_df = pd.DataFrame(questions)
        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            # Ensure existing_df has explanation column
            if "explanation" not in existing_df.columns:
                existing_df["explanation"] = ""
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_csv(path, index=False)
        print(f"Saved {len(combined_df)} questions to {path}:")
        print(combined_df.to_dict(orient='records'))  # Debug
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
        "answer": "Option A",
        "explanation": "Explanation text"
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
        print(f"Generated {len(questions)} questions: {questions}")  # Debug
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
    question_options = [5, 10, 15, 20, "Select All"]
    question_count = st.selectbox("How many questions do you want to attempt?", question_options, index=1)
    total_questions = len(df_full)
    if question_count == "Select All":
        question_count = total_questions
    else:
        question_count = min(question_count, max_qs)

    quiz_key = f"{grade}_{subject}_{lesson}_{'gen' if generated else 'orig'}"

    # Generate question ranges for "Select All"
    range_options = []
    if question_count == total_questions:
        for start in range(1, total_questions + 1, 5):
            end = min(start + 4, total_questions)
            range_options.append(f"Questions {start} to {end}")

    # Initialize or reset session state
    if quiz_key not in st.session_state or "df_full" not in st.session_state[quiz_key]:
        st.session_state[quiz_key] = {
            "current_q": 0,
            "answers": [""] * min(5, question_count) if question_count == total_questions else [""] * question_count,
            "submitted": False,
            "show_explanation": [False] * min(5, question_count) if question_count == total_questions else [
                                                                                                               False] * question_count,
            "start_index": 0,
            "set_index": 0,  # Used for non-Select All sets
            "df_full": df_full,  # Unshuffled for "Select All"
            "df_shuffled": df_full.sample(frac=1).reset_index(drop=True),  # Shuffled for other counts
            "selected_range": range_options[0] if range_options else f"Questions 1 to {min(5, total_questions)}",
        }
    # Reset answers and state if question_count changes
    elif len(st.session_state[quiz_key]["answers"]) != (
    min(5, question_count) if question_count == total_questions else question_count):
        st.session_state[quiz_key]["answers"] = [""] * min(5,
                                                           question_count) if question_count == total_questions else [
                                                                                                                         ""] * question_count
        st.session_state[quiz_key]["show_explanation"] = [False] * min(5,
                                                                       question_count) if question_count == total_questions else [
                                                                                                                                     False] * question_count
        st.session_state[quiz_key]["current_q"] = 0
        st.session_state[quiz_key]["submitted"] = False
        st.session_state[quiz_key]["start_index"] = 0
        st.session_state[quiz_key]["set_index"] = 0
        st.session_state[quiz_key]["df_shuffled"] = st.session_state[quiz_key]["df_full"].sample(frac=1).reset_index(
            drop=True)
        st.session_state[quiz_key]["selected_range"] = range_options[
            0] if range_options else f"Questions 1 to {min(5, total_questions)}"

    state = st.session_state[quiz_key]
    df_shuffled = state["df_shuffled"]
    df_full = state["df_full"]
    start_index = state["start_index"]
    set_index = state["set_index"]

    # Handle question range selection for "Select All"
    if question_count == total_questions and range_options:
        selected_range = st.selectbox("Select Question Range", range_options,
                                      index=range_options.index(
                                          state["selected_range"]) if "selected_range" in state and state[
                                          "selected_range"] in range_options else 0)
        if selected_range != state["selected_range"]:
            state["selected_range"] = selected_range
            state["current_q"] = 0
            state["submitted"] = False
            start_idx = int(selected_range.split("Questions ")[1].split(" to ")[0]) - 1
            end_idx = int(selected_range.split(" to ")[1])
            state["start_index"] = start_idx
            state["answers"] = [""] * (end_idx - start_idx)
            state["show_explanation"] = [False] * (end_idx - start_idx)
        start_idx = int(state["selected_range"].split("Questions ")[1].split(" to ")[0]) - 1
        end_idx = int(state["selected_range"].split(" to ")[1])
        set_size = end_idx - start_idx
        df = df_full.iloc[start_idx:end_idx].reset_index(drop=True)
        set_display = f"Questions {start_idx + 1} to {end_idx}"
    else:
        end_index = min(start_index + question_count, total_questions)
        df = df_shuffled.iloc[start_index:end_index].reset_index(drop=True)
        set_display = f"Questions {start_index + 1} to {end_index}"
        set_size = question_count

    if df.empty:
        st.warning("No more questions available. Starting over from the beginning.")
        state["start_index"] = 0
        state["set_index"] = 0
        if question_count != total_questions:
            state["df_shuffled"] = df_full.sample(frac=1).reset_index(drop=True)
        if question_count == total_questions and range_options:
            state["selected_range"] = range_options[0]
            start_idx = 0
            end_idx = min(5, total_questions)
            df = df_full.iloc[start_idx:end_idx].reset_index(drop=True)
            state["answers"] = [""] * (end_idx - start_idx)
            state["show_explanation"] = [False] * (end_idx - start_idx)
            set_display = f"Questions 1 to {end_idx}"
        else:
            df = df_shuffled.iloc[0:question_count].reset_index(drop=True)
            set_display = f"Questions 1 to {len(df)}"

    source = "Generated" if generated else "Original"
    st.subheader(f"{source} Quiz - {grade} - {subject} - {lesson.capitalize()} ({set_display})")

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

            st.markdown(f"---\n**Q{i + 1}: {q['question']}**")
            st.markdown(f"Your answer: {user_ans}  \n{result}")

            # Button to toggle explanation
            if st.button(f"Explain Answer Q{i + 1}", key=f"explain_{quiz_key}_{i}"):
                state["show_explanation"][i] = not state["show_explanation"][i]

            if state["show_explanation"][i]:
                with st.spinner(f"Getting explanation for Q{i + 1}..."):
                    explanation = get_gemini_explanation(
                        grade, subject,
                        q["question"],
                        [q["option1"], q["option2"], q["option3"], q["option4"]],
                        correct_ans,
                        user_ans,
                    )
                    st.markdown(f"**Explanation:**\n{explanation}")
                time.sleep(1)

        col1, col2 = st.columns(2)
        with col1:
            st.button("Restart Quiz", key=f"restart_{quiz_key}",
                      on_click=lambda: restart_quiz(quiz_key, question_count, total_questions, range_options))
        with col2:
            if question_count == total_questions and end_idx < total_questions and range_options:
                st.button("Next Range", key=f"next_set_{quiz_key}",
                          on_click=lambda: next_questions(quiz_key, question_count, total_questions, range_options))
            elif question_count != total_questions and end_index < total_questions:
                st.button(f"Next {question_count} Questions", key=f"next_set_{quiz_key}",
                          on_click=lambda: next_questions(quiz_key, question_count, total_questions, range_options))
            else:
                st.button("Start Over from Beginning", key=f"start_over_{quiz_key}",
                          on_click=lambda: next_questions(quiz_key, question_count, total_questions, range_options))
        return

    row = df.iloc[state["current_q"]]
    options = [str(row["option1"]), str(row["option2"]), str(row["option3"]), str(row["option4"])]
    # Debug: Log options and current answer
    print(f"Question {state['current_q'] + 1} options: {options}")
    print(f"Current answer in state: {state['answers'][state['current_q']]}")
    # Validate options to prevent IndexError
    if not options or not all(opt and isinstance(opt, str) for opt in options):
        st.error(f"Invalid question data for Q{state['current_q'] + 1} in '{lesson}.csv'. Options: {options}")
        return
    # Validate answer
    if str(row["answer"]).strip() not in options:
        st.error(
            f"Invalid answer for Q{state['current_q'] + 1} in '{lesson}.csv'. Answer: {row['answer']} not in options: {options}")
        return
    question = f"Q{state['current_q'] + 1}: {row['question']}"

    try:
        # Ensure current answer is valid
        current_answer = state["answers"][state["current_q"]]
        if current_answer and current_answer not in options:
            print(f"Resetting invalid answer for Q{state['current_q'] + 1}: {current_answer} not in {options}")
            state["answers"][state["current_q"]] = ""
            current_answer = ""
        user_choice = st.radio(
            question,
            options,
            key=f"{quiz_key}_{state['current_q']}",
            index=options.index(current_answer) if current_answer in options else 0,
        )
        state["answers"][state["current_q"]] = user_choice
    except ValueError as e:
        st.error(
            f"Error rendering question Q{state['current_q'] + 1}: {str(e)}. Options: {options}, Answer: {row['answer']}, Stored answer: {state['answers'][state['current_q']]}")
        return

    col1, col2 = st.columns(2)
    with col1:
        if state["current_q"] < len(df) - 1:
            st.button("Next", key=f"next_{quiz_key}_{state['current_q']}",
                      on_click=lambda: advance_question(1, len(df), quiz_key))
        else:
            st.button("Submit Set", key=f"submit_{quiz_key}", on_click=lambda: submit_quiz(quiz_key))

    with col2:
        if state["current_q"] > 0:
            st.button("Previous", key=f"prev_{quiz_key}_{state['current_q']}",
                      on_click=lambda: advance_question(-1, len(df), quiz_key))


def advance_question(delta, total, quiz_key):
    state = st.session_state[quiz_key]
    new_q = state["current_q"] + delta
    if 0 <= new_q < total:
        state["current_q"] = new_q
    elif new_q == total and state["current_q"] == total - 1:
        state["submitted"] = True  # Auto-submit after last question in set


def submit_quiz(quiz_key):
    st.session_state[quiz_key]["submitted"] = True


def restart_quiz(quiz_key, question_count, total_questions, range_options):
    state = st.session_state[quiz_key]
    set_size = min(5, question_count) if question_count == total_questions else question_count
    state["current_q"] = 0
    state["answers"] = [""] * set_size
    state["submitted"] = False
    state["show_explanation"] = [False] * set_size
    if question_count == total_questions and range_options:
        state["selected_range"] = range_options[0]


def next_questions(quiz_key, question_count, total_questions, range_options):
    state = st.session_state[quiz_key]
    if question_count == total_questions:
        current_range = state["selected_range"]
        current_idx = range_options.index(current_range) if current_range in range_options else 0
        next_idx = (current_idx + 1) % len(range_options) if range_options else 0
        state["selected_range"] = range_options[next_idx]
        state["start_index"] = int(state["selected_range"].split("Questions ")[1].split(" to ")[0]) - 1
        set_size = int(state["selected_range"].split(" to ")[1]) - state["start_index"]
        state["current_q"] = 0
        state["answers"] = [""] * set_size
        state["submitted"] = False
        state["show_explanation"] = [False] * set_size
    else:
        state["start_index"] += question_count
        if state["start_index"] >= total_questions:
            state["start_index"] = 0
            state["df_shuffled"] = state["df_full"].sample(frac=1).reset_index(drop=True)  # Re-shuffle
        state["current_q"] = 0
        state["answers"] = [""] * question_count
        state["submitted"] = False
        state["show_explanation"] = [False] * question_count


# === Main App Logic ===
if not check_login():
    st.stop()  # Stop rendering if not logged in

st.sidebar.title("Navigation")
main_page = st.sidebar.radio("Go to", ["About", "Subjects", "Generate Extra Questions"],
                             index=1)  # Default to "Subjects"

if main_page == "About":
    st.subheader("📘 About This Quiz App")
    st.markdown("""
- This app is designed for Grade 5 and Grade 7 students.
- Choose your grade (Grade 5 or Grade 7), a subject (e.g., Math), and a lesson (e.g., Roots) to begin.
- Answer multiple-choice questions and get instant feedback.
- After submitting, view results and click "Explain Answer" for each question to see explanations powered by **Gemini AI**.
- Use the "Next Questions" button to take the next set of questions, or "Restart Quiz" to redo the current set.
- For "Select All", choose a specific range of questions (e.g., Questions 1 to 5) in their original order.
- You can also generate extra questions using AI to practice more.
""")

elif main_page == "Subjects":
    subjects = get_subjects()
    if not subjects:
        st.error("No grades found. Please create input/Grade5 and/or input/Grade7 folders.")
    else:
        grade = st.sidebar.selectbox("Choose Grade", subjects,
                                     index=subjects.index("Grade7") if "Grade7" in subjects else 0)
        grade_subjects = get_subjects_in_grade(grade)
        if grade_subjects:
            subject = st.sidebar.selectbox("Choose Subject", grade_subjects,
                                           index=grade_subjects.index("Math") if "Math" in grade_subjects else 0)
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
            subject = st.selectbox("Choose Subject", grade_subjects,
                                   index=grade_subjects.index("Math") if "Math" in grade_subjects else 0)
            lessons = get_lessons(grade, subject)
            if lessons:
                lesson = st.selectbox("Choose Lesson", lessons)
                num_questions = st.number_input("Number of Questions to Generate", min_value=1, max_value=10, value=5)
                if st.button("Generate Questions"):
                    with st.spinner("Generating questions with Gemini AI..."):
                        questions = generate_questions(grade, subject, lesson, num_questions)
                        if questions:
                            save_generated_questions(grade, subject, lesson, questions)
                            st.success(
                                f"✅ {num_questions} questions generated and saved for {grade} - {subject} - {lesson}!")
                            st.markdown(
                                "You can now take the generated quiz by selecting 'Generated' under Quiz Type in the Subjects page.")
                        else:
                            st.error("Failed to generate questions. Please try again.")
            else:
                st.warning(f"No lessons (CSV files) found in 'input/{grade}/{subject}/'")
        else:
            st.warning(f"No subjects found in 'input/{grade}/'. Please add subject folders (e.g., Math).")