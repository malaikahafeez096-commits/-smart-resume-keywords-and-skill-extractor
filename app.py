
import streamlit as st
import joblib
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="Smart Resume Keywords & Skill Extractor (ATS)",
    page_icon="📄",
    layout="wide"
)

# Load Trained Model and Vectorizer
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('ats_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_artifacts()

# App Title & Description
st.title("📄 Smart Resume Keywords & Skill Extractor (ATS)")
st.markdown("Evaluate resume compatibility against job descriptions using TF-IDF Vectorization & Skill Gap Analysis (No external APIs used).")

if vectorizer is None:
    st.error("⚠️ Model files (`ats_model.pkl` and `tfidf_vectorizer.pkl`) not found in your project directory! Please run your Jupyter Notebook training script first.")
else:
    # Layout: Two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Candidate Resume")
        resume_input = st.text_area(
            "Paste Resume Text Here:",
            height=250,
            placeholder="Experienced Data Scientist skilled in Python, Machine Learning, SQL..."
        )

    with col2:
        st.subheader("📋 Job Description (JD)")
        jd_input = st.text_area(
            "Paste Job Description Text Here:",
            height=250,
            placeholder="Looking for a Data Scientist with expertise in Python, Pandas, AWS..."
        )

    # Updated Skill Synonym Dictionary including shorthand like 'sklearn', 'py', 'ml'
    skill_synonyms = {
        'Python': ['python', 'py'],
        'Machine Learning': ['machine learning', 'ml'],
        'SQL': ['sql', 'postgresql', 'postgres'],
        'Data Analysis': ['data analysis', 'data analytics'],
        'Pandas': ['pandas'],
        'TensorFlow': ['tensorflow', 'tf'],
        'Scikit-Learn': ['scikit-learn', 'sklearn'],
        'Docker': ['docker'],
        'AWS': ['aws'],
        'FastAPI': ['fastapi'],
        'Git': ['git'],
        'Java': ['java'],
        'C++': ['c++', 'cpp'],
        'Tableau': ['tableau'],
        'JavaScript': ['javascript', 'js'],
        'React': ['react', 'reactjs']
    }

    skills_pool = list(skill_synonyms.keys())

    # Predict Button
    if st.button("Evaluate Match & Extract Skills", type="primary"):
        if not resume_input.strip() or not jd_input.strip():
            st.warning("Please provide both a resume and a job description text.")
        else:
            # 1. Separate Skill Extraction Function using Synonyms & Word Boundaries
            def extract_mapped_skills(text):
                found_skills = set()
                for standard_skill, variants in skill_synonyms.items():
                    for variant in variants:
                        pattern = r'\b' + re.escape(variant) + r'\b'
                        if re.search(pattern, text, re.IGNORECASE):
                            found_skills.add(standard_skill)
                            break
                return found_skills

            resume_skills_found = extract_mapped_skills(resume_input)
            jd_skills_found = extract_mapped_skills(jd_input)

            # Intersection = Matching Skills, Difference = Missing Skills
            matched_skills = list(resume_skills_found.intersection(jd_skills_found))
            missing_skills = list(jd_skills_found - resume_skills_found)

            # 2. Robust Scoring Engine
            if len(jd_skills_found) > 0:
                skill_match_ratio = len(matched_skills) / len(jd_skills_found)
            else:
                skill_match_ratio = 0.3

            # Calculate semantic similarity
            resume_vec = vectorizer.transform([resume_input])
            jd_vec = vectorizer.transform([jd_input])
            similarity_score = cosine_similarity(resume_vec, jd_vec)[0][0]

            # GUARANTEED HIGH SCORE FOR PERFECT MATCHES:
            if len(missing_skills) == 0 and len(jd_skills_found) > 0:
                final_score = 85.0 + (len(matched_skills) * 1.5)
                final_score = min(98.0, final_score) # Cap at 98%
            else:
                final_score = (skill_match_ratio * 0.70 + similarity_score * 0.30) * 100

            final_score = max(5.0, min(98.0, final_score))
            is_shortlisted = final_score >= 50.0

            st.markdown("---")
            st.subheader("📊 ATS Evaluation Results")

            # Metrics Display
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Match Probability", value=f"{final_score:.1f}%")
            with m_col2:
                status = "Shortlisted / High Match" if is_shortlisted else "Rejected / Low Match"
                st.metric(label="ATS Recommendation", value=status)

            # Progress Bar Visualization
            st.progress(float(final_score / 100))

            # 3. Display Skill Gap Analysis
            st.markdown("---")
            st.subheader("🔍 Keyword & Skill Gap Analysis")

            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("### ✅ Matching Skills Found")
                if matched_skills:
                    for s in sorted(matched_skills):
                        st.write(f"- **{s}**")
                else:
                    st.write("No overlapping key skills detected from the standard pool.")

            with sc2:
                st.markdown("### ❌ Missing / Recommended Skills")
                if missing_skills:
                    for s in sorted(missing_skills):
                        st.write(f"- **{s}**")
                else:
                    st.write("Great! All required skills mentioned in the job description are present in the resume.")
