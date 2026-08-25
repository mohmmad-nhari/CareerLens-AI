import re
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util


# =========================================================
# PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(58, 123, 213, 0.12), transparent 30%),
            radial-gradient(circle at top right, rgba(111, 66, 193, 0.10), transparent 25%),
            #0b0f19;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .hero {
        padding: 38px;
        border-radius: 26px;
        background: linear-gradient(
            135deg,
            rgba(35, 45, 75, 0.95),
            rgba(18, 24, 42, 0.95)
        );
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 8px;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #b8c0d4;
        max-width: 820px;
    }

    .glass-card {
        padding: 24px;
        border-radius: 22px;
        background: rgba(20, 27, 45, 0.86);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 35px rgba(0,0,0,0.25);
        margin-bottom: 18px;
    }

    .score-card {
        text-align: center;
        padding: 22px;
        border-radius: 20px;
        background: linear-gradient(
            145deg,
            rgba(32, 40, 66, 0.95),
            rgba(18, 23, 38, 0.95)
        );
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 150px;
    }

    .score-label {
        color: #98a2b8;
        font-size: 15px;
        margin-bottom: 10px;
    }

    .score-value {
        font-size: 42px;
        font-weight: 800;
        line-height: 1;
    }

    .score-good {
        color: #39d98a;
    }

    .score-mid {
        color: #ffcc66;
    }

    .score-low {
        color: #ff6b6b;
    }

    .section-title {
        font-size: 26px;
        font-weight: 750;
        margin-bottom: 14px;
        margin-top: 6px;
    }

    .skill-good {
        display: inline-block;
        background: rgba(57, 217, 138, 0.14);
        color: #6ee7b7;
        padding: 8px 12px;
        border-radius: 999px;
        margin: 5px 5px 5px 0;
        border: 1px solid rgba(57, 217, 138, 0.25);
        font-size: 14px;
    }

    .skill-missing {
        display: inline-block;
        background: rgba(255, 107, 107, 0.12);
        color: #ff9292;
        padding: 8px 12px;
        border-radius: 999px;
        margin: 5px 5px 5px 0;
        border: 1px solid rgba(255, 107, 107, 0.24);
        font-size: 14px;
    }

    .check-item {
        padding: 10px 12px;
        margin-bottom: 8px;
        border-radius: 12px;
        background: rgba(255,255,255,0.035);
    }

    .recommendation {
        padding: 14px 16px;
        border-radius: 14px;
        background: rgba(99, 102, 241, 0.09);
        border-left: 4px solid #7c83ff;
        margin-bottom: 10px;
        color: #e3e8f3;
    }

    .small-muted {
        color: #8c96aa;
        font-size: 13px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 18px;
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 16px !important;
    }

    .stButton > button {
        border-radius: 14px;
        padding: 0.75rem 1.3rem;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: #0f1422;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def score_class(score):
    if score >= 75:
        return "score-good"
    elif score >= 50:
        return "score-mid"
    return "score-low"


def render_score_card(title, value, icon):
    color_class = score_class(value)

    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">{icon} {title}</div>
            <div class="score-value {color_class}">
                {value:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# LOAD AI MODEL
# =========================================================
@st.cache_resource
def load_ai_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_ai_model()


# =========================================================
# PDF
# =========================================================
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# SKILLS DATABASE
# =========================================================
skills_list = [
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "html",
    "css",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "flask",
    "django",
    "fastapi",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data analysis",
    "data science",
    "opencv",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "git",
    "github",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "linux",
    "raspberry pi",
    "robotics",
    "unity",
    "database design",
    "software testing",
    "problem solving",
    "communication",
    "teamwork"
]


def find_skills(text):
    text = text.lower()
    found_skills = []

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills


# =========================================================
# SEMANTIC AI
# =========================================================
def calculate_semantic_similarity(
    resume_text,
    job_description
):
    resume_embedding = model.encode(
        resume_text,
        convert_to_tensor=True
    )

    job_embedding = model.encode(
        job_description,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    ).item()

    similarity = max(
        0,
        min(similarity, 1)
    )

    return similarity * 100


def calculate_overall_score(
    skills_score,
    semantic_score
):
    return (
        skills_score * 0.60
        +
        semantic_score * 0.40
    )


# =========================================================
# ATS
# =========================================================
def calculate_ats_readiness(
    resume_text,
    job_skills,
    matched_skills
):
    text = resume_text.lower()

    ats_score = 0
    checks = []

    email_found = re.search(
        r"[\w\.-]+@[\w\.-]+\.\w+",
        resume_text
    )

    if email_found:
        ats_score += 10
        checks.append(
            ("pass", "Email address detected.")
        )
    else:
        checks.append(
            ("fail", "No email address detected.")
        )

    phone_found = re.search(
        r"(\+?\d[\d\s\-\(\)]{7,}\d)",
        resume_text
    )

    if phone_found:
        ats_score += 10
        checks.append(
            ("pass", "Phone number detected.")
        )
    else:
        checks.append(
            ("fail", "No phone number detected.")
        )

    education_keywords = [
        "education",
        "university",
        "college",
        "bachelor",
        "degree"
    ]

    education_found = any(
        keyword in text
        for keyword in education_keywords
    )

    if education_found:
        ats_score += 10
        checks.append(
            ("pass", "Education information detected.")
        )
    else:
        checks.append(
            ("fail", "Education section may be missing.")
        )

    if (
        "skills" in text
        or len(find_skills(resume_text)) >= 3
    ):
        ats_score += 10
        checks.append(
            ("pass", "Skills information detected.")
        )
    else:
        checks.append(
            ("fail", "A clear skills section is recommended.")
        )

    experience_keywords = [
        "experience",
        "employment",
        "work experience",
        "projects",
        "project",
        "internship"
    ]

    experience_found = any(
        keyword in text
        for keyword in experience_keywords
    )

    if experience_found:
        ats_score += 15
        checks.append(
            (
                "pass",
                "Experience or project information detected."
            )
        )
    else:
        checks.append(
            (
                "fail",
                "Consider adding experience, internships, or projects."
            )
        )

    if job_skills:
        coverage = (
            len(matched_skills)
            /
            len(job_skills)
        )

        ats_score += coverage * 30

        checks.append(
            (
                "info",
                f"Job skill coverage: {coverage * 100:.0f}%."
            )
        )

    word_count = len(
        resume_text.split()
    )

    if 200 <= word_count <= 1200:
        ats_score += 15
        checks.append(
            (
                "pass",
                f"Resume length looks reasonable ({word_count} words)."
            )
        )

    elif word_count < 200:
        ats_score += 5
        checks.append(
            (
                "warning",
                f"Resume may be too short ({word_count} words)."
            )
        )

    else:
        ats_score += 8
        checks.append(
            (
                "warning",
                f"Resume may be too long ({word_count} words)."
            )
        )

    return min(ats_score, 100), checks


# =========================================================
# RESUME SECTIONS
# =========================================================
def analyze_resume_sections(resume_text):
    text = resume_text.lower()

    sections = []

    email_found = re.search(
        r"[\w\.-]+@[\w\.-]+\.\w+",
        resume_text
    )

    phone_found = re.search(
        r"(\+?\d[\d\s\-\(\)]{7,}\d)",
        resume_text
    )

    sections.append(
        (
            "Contact Information",
            bool(email_found or phone_found)
        )
    )

    summary_keywords = [
        "summary",
        "professional summary",
        "profile",
        "objective",
        "career objective"
    ]

    sections.append(
        (
            "Professional Summary",
            any(
                keyword in text
                for keyword in summary_keywords
            )
        )
    )

    education_keywords = [
        "education",
        "university",
        "college",
        "bachelor",
        "degree"
    ]

    sections.append(
        (
            "Education",
            any(
                keyword in text
                for keyword in education_keywords
            )
        )
    )

    sections.append(
        (
            "Technical Skills",
            (
                "skills" in text
                or len(find_skills(resume_text)) >= 3
            )
        )
    )

    project_keywords = [
        "project",
        "projects",
        "graduation project"
    ]

    sections.append(
        (
            "Projects",
            any(
                keyword in text
                for keyword in project_keywords
            )
        )
    )

    experience_keywords = [
        "work experience",
        "experience",
        "employment",
        "internship"
    ]

    sections.append(
        (
            "Work Experience",
            any(
                keyword in text
                for keyword in experience_keywords
            )
        )
    )

    certification_keywords = [
        "certification",
        "certifications",
        "certificate",
        "certificates",
        "course",
        "courses",
        "academy"
    ]

    sections.append(
        (
            "Certifications / Courses",
            any(
                keyword in text
                for keyword in certification_keywords
            )
        )
    )

    language_keywords = [
        "languages",
        "language",
        "arabic",
        "english"
    ]

    sections.append(
        (
            "Languages",
            any(
                keyword in text
                for keyword in language_keywords
            )
        )
    )

    return sections


# =========================================================
# RECOMMENDATIONS
# =========================================================
def generate_recommendations(
    matched_skills,
    missing_skills,
    overall_score
):
    recommendations = []

    if overall_score >= 75:
        recommendations.append(
            "Your resume is a strong match for this role."
        )

    elif overall_score >= 50:
        recommendations.append(
            "Your resume has a moderate match for this role."
        )

    else:
        recommendations.append(
            "Your resume currently has a low match for this role."
        )

    if matched_skills:
        matched_text = ", ".join(
            skill.title()
            for skill in matched_skills
        )

        recommendations.append(
            f"Strong matches: {matched_text}."
        )

    if missing_skills:
        missing_text = ", ".join(
            skill.title()
            for skill in missing_skills
        )

        recommendations.append(
            f"Missing or unclear skills: {missing_text}."
        )

        recommendations.append(
            "Highlight these skills only if you genuinely have experience with them."
        )

    return recommendations


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🤖 AI Resume Matcher")

    st.caption(
        "Smart resume analysis powered by semantic AI."
    )

    st.divider()

    st.markdown("### What it analyzes")

    st.write("🎯 Job Skill Match")
    st.write("🧠 Semantic Similarity")
    st.write("📊 ATS Readiness")
    st.write("📋 Resume Sections")
    st.write("💡 Recommendations")

    st.divider()

    st.caption(
        "All scores are estimates and should not be treated "
        "as employer decisions."
    )


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🤖 AI Resume Matcher
        </div>
        <div class="hero-subtitle">
            Analyze your resume against a job description using
            semantic AI, skill matching, ATS readiness checks,
            and actionable recommendations.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INPUT AREA
# =========================================================
left_input, right_input = st.columns(
    [1, 1.4],
    gap="large"
)


with left_input:

    st.markdown(
        '<div class="section-title">📄 Resume</div>',
        unsafe_allow_html=True
    )

    uploaded_cv = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.caption(
        "PDF files only. Text-based PDFs work best."
    )


with right_input:

    st.markdown(
        '<div class="section-title">💼 Job Description</div>',
        unsafe_allow_html=True
    )

    job_description = st.text_area(
        "Job Description",
        height=220,
        placeholder=(
            "Paste the full job description here..."
        ),
        label_visibility="collapsed"
    )


analyze_button = st.button(
    "✨ Analyze Resume",
    type="primary",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================
if analyze_button:

    if uploaded_cv is None:
        st.warning(
            "Please upload a resume PDF."
        )

    elif not job_description.strip():
        st.warning(
            "Please paste a job description."
        )

    else:

        with st.spinner(
            "AI is analyzing your resume..."
        ):

            try:
                resume_text = (
                    extract_text_from_pdf(
                        uploaded_cv
                    )
                )

            except Exception:
                st.error(
                    "The PDF could not be read."
                )
                st.stop()

            if not resume_text.strip():
                st.error(
                    "No readable text was found in the PDF."
                )
                st.stop()

            resume_skills = find_skills(
                resume_text
            )

            job_skills = find_skills(
                job_description
            )

            matched_skills = [
                skill
                for skill in job_skills
                if skill in resume_skills
            ]

            missing_skills = [
                skill
                for skill in job_skills
                if skill not in resume_skills
            ]

            if job_skills:
                skills_score = (
                    len(matched_skills)
                    /
                    len(job_skills)
                ) * 100
            else:
                skills_score = 0

            semantic_score = (
                calculate_semantic_similarity(
                    resume_text,
                    job_description
                )
            )

            overall_score = (
                calculate_overall_score(
                    skills_score,
                    semantic_score
                )
            )

            ats_score, ats_checks = (
                calculate_ats_readiness(
                    resume_text,
                    job_skills,
                    matched_skills
                )
            )

            resume_sections = (
                analyze_resume_sections(
                    resume_text
                )
            )

            recommendations = (
                generate_recommendations(
                    matched_skills,
                    missing_skills,
                    overall_score
                )
            )


        st.success(
            "Analysis completed successfully."
        )


        # =================================================
        # SCORE CARDS
        # =================================================
        st.markdown(
            '<div class="section-title">📊 Analysis Overview</div>',
            unsafe_allow_html=True
        )

        score1, score2, score3, score4 = st.columns(
            4,
            gap="medium"
        )

        with score1:
            render_score_card(
                "Overall Match",
                overall_score,
                "🏆"
            )

        with score2:
            render_score_card(
                "Skills Match",
                skills_score,
                "🎯"
            )

        with score3:
            render_score_card(
                "Semantic AI",
                semantic_score,
                "🧠"
            )

        with score4:
            render_score_card(
                "ATS Readiness",
                ats_score,
                "📈"
            )


        st.markdown("<br>", unsafe_allow_html=True)


        # =================================================
        # MATCHED / MISSING
        # =================================================
        skill_col1, skill_col2 = st.columns(
            2,
            gap="large"
        )

        with skill_col1:

            st.markdown(
                '<div class="section-title">✅ Matched Skills</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="glass-card">',
                unsafe_allow_html=True
            )

            if matched_skills:

                skill_html = ""

                for skill in matched_skills:
                    skill_html += (
                        f'<span class="skill-good">'
                        f'✓ {skill.title()}'
                        f'</span>'
                    )

                st.markdown(
                    skill_html,
                    unsafe_allow_html=True
                )

            else:
                st.write(
                    "No matched skills detected."
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        with skill_col2:

            st.markdown(
                '<div class="section-title">❌ Missing Skills</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="glass-card">',
                unsafe_allow_html=True
            )

            if missing_skills:

                skill_html = ""

                for skill in missing_skills:
                    skill_html += (
                        f'<span class="skill-missing">'
                        f'• {skill.title()}'
                        f'</span>'
                    )

                st.markdown(
                    skill_html,
                    unsafe_allow_html=True
                )

            else:
                st.write(
                    "No missing skills detected."
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # =================================================
        # ATS
        # =================================================
        st.markdown(
            '<div class="section-title">📊 ATS Readiness Analysis</div>',
            unsafe_allow_html=True
        )

        st.progress(
            int(ats_score)
        )

        st.caption(
            "Estimated readiness only — not an employer ATS score."
        )

        with st.expander(
            "View detailed ATS checks"
        ):

            for status, message in ats_checks:

                if status == "pass":
                    icon = "✅"

                elif status == "fail":
                    icon = "❌"

                elif status == "warning":
                    icon = "⚠️"

                else:
                    icon = "ℹ️"

                st.markdown(
                    f"""
                    <div class="check-item">
                        {icon} {message}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # =================================================
        # RESUME SECTIONS
        # =================================================
        st.markdown(
            '<div class="section-title">📋 Resume Structure</div>',
            unsafe_allow_html=True
        )

        section_cols = st.columns(2)

        for index, (
            section_name,
            found
        ) in enumerate(resume_sections):

            target_col = section_cols[
                index % 2
            ]

            with target_col:

                if found:
                    st.markdown(
                        f"""
                        <div class="check-item">
                            ✅ {section_name}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.markdown(
                        f"""
                        <div class="check-item">
                            ⚠️ {section_name}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # =================================================
        # RECOMMENDATIONS
        # =================================================
        st.markdown(
            '<div class="section-title">💡 Smart Recommendations</div>',
            unsafe_allow_html=True
        )

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="recommendation">
                    {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # DETAILS
        # =================================================
        with st.expander(
            "📄 View extracted resume text"
        ):

            st.text_area(
                "Resume Text",
                resume_text,
                height=350,
                label_visibility="collapsed"
            )