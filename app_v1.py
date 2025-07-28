import streamlit as st
import pandas as pd
import os
import time
import google.generativeai as genai

# === Configure Gemini API ===
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not GEMINI_API_KEY:
    st.error("Please add your Google Gemini API key to the Streamlit secrets.toml file or environment variables.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# === Streamlit Config ===
st.set_page_config(page_title="Quiz App with Gemini AI", layout="centered")

# === Paths ===
BASE_PATH = "input"
GEN_PATH = "generated"

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
main_page = st.sidebar.radio("Go to", ["About", "Subjects", "Generate Extra Questions"])

# === Utility Functions ===
def get_subjects():
    return sorted([d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))])

def get_lessons(subject):
    folder = os.path.join(BASE_PATH, subject)
    return [f[:-4] for f in os.listdir(folder) if f.endswith(".csv")]

def load_quiz(subject, lesson):
    path = os.path.join(BASE_PATH, subject, f"{lesson}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_generated_quiz(subject, lesson):
    path = os.path.join(GEN_PATH, subject, f"{lesson}_generated.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def save_generated_questions(subject, lesson, questions):
    gen_dir = os.path.join(GEN_PATH, subject)
    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, f"{lesson}_generated.csv")

    new_df = pd.DataFrame(questions)
    if os.path.exists(path):
        existing_df = pd.read_csv(path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(path, index=False)

@st.cache_data(show_spinner=False)
def get_gemini_explanation(question, options, correct_answer, user_answer):
    prompt = f"""
You are a helpful tutor. Explain this multiple-choice question in a student-friendly way.

Question: {question}
Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

The correct answer is: {correct_answer}
The student's answer was: {user_answer}

Explain why the correct answer is correct, and if the student was wrong, clarify the mistake.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Could not fetch explanation: {str(e)}"

def run_quiz(subject, lesson, generated=False):
    df_full = load_generated_quiz(subject, lesson) if generated else load_quiz(subject, lesson)
    if df_full is None:
        st.warning("Quiz data not found.")
        return

    max_qs = min(20, len(df_full))
    question_count = st.selectbox("How many questions do you want to attempt?", [5, 10, 15, 20], index=1)
    question_count = min(question_count, max_qs)

    df = df_full.sample(n=question_count, random_state=42).reset_index(drop=True)

    source = "Generated" if generated else "Original"
    st.subheader(f"{source} Quiz - {subject.capitalize()} - {lesson.capitalize()}")

    quiz_key = f"{subject}_{lesson}_{'gen' if generated else 'orig'}"
    if quiz_key not in st.session_state:
        st.session_state[quiz_key] = {
            "current_q": 0,
            "answers": [""] * len(df),
            "submitted": False,
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

            with st.spinner("Getting explanation from Gemini..."):
                explanation = get_gemini_explanation(
                    q["question"],
                    [q["option1"], q["option2"], q["option3"], q["option4"]],
                    correct_ans,
                    user_ans,
                )
                st.markdown(f"**Explanation:**\n{explanation}")
            time.sleep(1)

        st.button("Restart Quiz", on_click=lambda: st.session_state.pop(quiz_key))
        return

    row = df.iloc[current_q]
    options = [row["option1"], row["option2"], row["option3"], row["option4"]]
    question = f"Q{current_q + 1}: {row['question']}"

    user_choice = st.radio(
        question,
        options,
        key=f"{quiz_key}_{current_q}",
        index=options.index(state["answers"][current_q]) if state["answers"][current_q] in options else 0,
    )
    state["answers"][current_q] = user_choice

    col1, col2 = st.columns(2)
    with col1:
        if current_q < len(df) - 1:
            st.button("Next", on_click=lambda: advance_question(1, len(df), quiz_key))
        else:
            st.button("Submit Quiz", on_click=lambda: submit_quiz(quiz_key))

    with col2:
        if current_q > 0:
            st.button("Previous", on_click=lambda: advance_question(-1, len(df), quiz_key))

def advance_question(delta, total, quiz_key):
    state = st.session_state[quiz_key]
    new_q = state["current_q"] + delta
    if 0 <= new_q < total:
        state["current_q"] = new_q

def submit_quiz(quiz_key):
    st.session_state[quiz_key]["submitted"] = True

# === About Page ===
if main_page == "About":
    st.subheader("📘 About This App")
    st.markdown("""
- Choose a subject and a lesson to begin.
- Answer multiple choice questions.
- After submitting, you'll receive explanations powered by **Gemini AI** to deepen your understanding!
""")

# === Subjects Page ===
elif main_page == "Subjects":
    subjects = get_subjects()
    if not subjects:
        st.error("No subjects found in 'input/' directory.")
    else:
        subject = st.sidebar.selectbox("Choose Subject", subjects)
        lessons = get_lessons(subject)
        if lessons:
            lesson = st.sidebar.selectbox("Choose Lesson", lessons)
            quiz_type = st.sidebar.radio("Quiz Type", ["Original", "Generated"])
            run_quiz(subject, lesson, generated=(quiz_type == "Generated"))
        else:
            st.warning(f"No lessons (CSV files) found in 'input/{subject}/'")

# === Generate Extra Questions ===
elif main_page == "Generate Extra Questions":
    st.subheader("🚧 Coming soon: AI-powered question generation.")
