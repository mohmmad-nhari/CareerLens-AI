import re
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util


# =========================================================
# PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="CareerLens AI",
    page_icon="🔮",
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

    .score-good { color: #39d98a; }
    .score-mid  { color: #ffcc66; }
    .score-low  { color: #ff6b6b; }

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

    .skill-partial {
        display: inline-block;
        background: rgba(255, 204, 102, 0.12);
        color: #ffd67a;
        padding: 8px 12px;
        border-radius: 999px;
        margin: 5px 5px 5px 0;
        border: 1px solid rgba(255, 204, 102, 0.28);
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
# HELPERS
# =========================================================
def score_class(score):
    if score >= 75:
        return "score-good"
    if score >= 50:
        return "score-mid"
    return "score-low"


def render_score_card(title, value, icon):
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">{icon} {title}</div>
            <div class="score-value {score_class(value)}">{value:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# LOAD AI MODEL
# =========================================================
@st.cache_resource
def load_ai_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model = load_ai_model()

model = load_ai_model()


# =========================================================
# PDF EXTRACTION
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

    "cybersecurity",
    "information security",
    "soc analyst",
    "security operations center",
    "network security",
    "incident response",
    "threat analysis",
    "vulnerability assessment",
    "siem",
    "splunk",
    "firewall",
    "intrusion detection",
    "penetration testing",
    "ethical hacking",
    "network monitoring",

    "database design",
    "software testing",
    "problem solving",
    "communication",
    "teamwork"
]
skill_aliases = {
    "artificial intelligence": [
        "artificial intelligence", "ai",
        "الذكاء الاصطناعي"
    ],

    "machine learning": [
        "machine learning", "ml",
        "تعلم الآلة", "التعلم الآلي"
    ],

    "data analysis": [
        "data analysis",
        "تحليل البيانات"
    ],

    "data science": [
        "data science",
        "علم البيانات"
    ],

    "javascript": ["javascript", "java script", "js"],
    "typescript": ["typescript", "type script", "ts"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "tensorflow": ["tensorflow", "tensor flow"],
    "pytorch": ["pytorch", "py torch"],
    "opencv": ["opencv", "open cv", "cv2"],
    "github": ["github", "git hub"],
    "raspberry pi": ["raspberry pi", "raspberrypi"],
    "sql": ["sql", "structured query language"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp"],

    "cybersecurity": [
        "cybersecurity",
        "cyber security",
        "الأمن السيبراني",
        "امن سيبراني"
    ],

    "information security": [
        "information security",
        "infosec",
        "أمن المعلومات",
        "امن المعلومات"
    ],

    "soc analyst": [
        "soc analyst",
        "security operations analyst",
        "محلل soc",
        "محلل مركز العمليات الأمنية",
        "محلل مركز العمليات الامنية"
    ],

    "security operations center": [
        "security operations center",
        "soc",
        "مركز العمليات الأمنية",
        "مركز العمليات الامنية"
    ],

    "network security": [
        "network security",
        "أمن الشبكات",
        "امن الشبكات"
    ],

    "incident response": [
        "incident response",
        "الاستجابة للحوادث",
        "استجابة للحوادث"
    ],

    "threat analysis": [
        "threat analysis",
        "threat intelligence",
        "تحليل التهديدات",
        "استخبارات التهديدات"
    ],

    "vulnerability assessment": [
        "vulnerability assessment",
        "vulnerability analysis",
        "تقييم الثغرات",
        "تحليل الثغرات"
    ],

    "siem": [
        "siem",
        "security information and event management",
        "إدارة معلومات وأحداث الأمن",
        "ادارة معلومات واحداث الامن"
    ],

    "splunk": ["splunk"],

    "firewall": [
        "firewall",
        "firewalls",
        "جدار الحماية",
        "جدران الحماية"
    ],

    "intrusion detection": [
        "intrusion detection",
        "ids",
        "ips",
        "كشف التسلل",
        "منع التسلل"
    ],

    "penetration testing": [
        "penetration testing",
        "pentesting",
        "pentest",
        "اختبار الاختراق"
    ],

    "ethical hacking": [
        "ethical hacking",
        "الاختراق الأخلاقي",
        "الاختراق الاخلاقي"
    ],

    "network monitoring": [
        "network monitoring",
        "مراقبة الشبكات",
        "مراقبة الشبكة"
    ],

    "problem solving": [
        "problem solving",
        "حل المشكلات",
        "حل المشاكل"
    ],

    "communication": [
        "communication",
        "communication skills",
        "مهارات التواصل",
        "التواصل"
    ],

    "teamwork": [
        "teamwork",
        "team work",
        "العمل الجماعي",
        "العمل ضمن فريق"
    ]
}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_skill(text, term):
    term = term.lower().strip()
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text) is not None


def find_skills(text):
    text = normalize_text(text)
    found_skills = []

    for skill in skills_list:
        aliases = skill_aliases.get(skill, [skill])
        if any(contains_skill(text, alias) for alias in aliases):
            found_skills.append(skill)

    return found_skills


def classify_skill_matches(job_skills, resume_skills):
    """
    Classify required job skills into:
    - exact matches
    - partial/related matches
    - missing skills
    """
    matched_skills = []
    partial_matches = []
    missing_skills = []

    related_skills = {
        "github": ["git"],
        "git": ["github"],
        "tensorflow": ["machine learning", "deep learning"],
        "pytorch": ["machine learning", "deep learning"],
        "scikit-learn": ["machine learning"],
        "opencv": ["artificial intelligence"],
        "flask": ["python"],
        "django": ["python"],
        "fastapi": ["python"],
        "mysql": ["sql"],
        "postgresql": ["sql"],
        "pandas": ["python", "data analysis"],
        "numpy": ["python", "data analysis"],
        "raspberry pi": ["robotics"],
    }

    for skill in job_skills:
        if skill in resume_skills:
            matched_skills.append(skill)
            continue

        related = related_skills.get(skill, [])
        if any(related_skill in resume_skills for related_skill in related):
            partial_matches.append(skill)
        else:
            missing_skills.append(skill)

    return matched_skills, partial_matches, missing_skills


# =========================================================
# SEMANTIC AI
# =========================================================
def calculate_semantic_similarity(resume_text, job_description):
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

    similarity = max(0, min(similarity, 1))
    return similarity * 100


def calculate_overall_score(skills_score, semantic_score):
    return (skills_score * 0.60) + (semantic_score * 0.40)


# =========================================================
# ATS ANALYSIS
# =========================================================
def calculate_ats_readiness(
    resume_text,
    job_skills,
    matched_skills,
    partial_matches
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
        checks.append(("pass", "Email address detected."))
    else:
        checks.append(("fail", "No email address detected."))

    phone_found = re.search(
        r"(\+?\d[\d\s\-\(\)]{7,}\d)",
        resume_text
    )

    if phone_found:
        ats_score += 10
        checks.append(("pass", "Phone number detected."))
    else:
        checks.append(("fail", "No phone number detected."))

    education_keywords = [
        "education",
        "university",
        "college",
        "bachelor",
        "degree",
    ]
    education_found = any(
        keyword in text
        for keyword in education_keywords
    )

    if education_found:
        ats_score += 10
        checks.append(("pass", "Education information detected."))
    else:
        checks.append(("fail", "Education section may be missing."))

    if "skills" in text or len(find_skills(resume_text)) >= 3:
        ats_score += 10
        checks.append(("pass", "Skills information detected."))
    else:
        checks.append(("fail", "A clear skills section is recommended."))

    experience_keywords = [
        "experience",
        "employment",
        "work experience",
        "projects",
        "project",
        "internship",
    ]
    experience_found = any(
        keyword in text
        for keyword in experience_keywords
    )

    if experience_found:
        ats_score += 15
        checks.append(("pass", "Experience or project information detected."))
    else:
        checks.append(
            ("fail", "Consider adding experience, internships, or projects.")
        )

    if job_skills:
        weighted_coverage = (
            len(matched_skills)
            + (len(partial_matches) * 0.5)
        ) / len(job_skills)

        ats_score += weighted_coverage * 30
        checks.append(
            (
                "info",
                f"Weighted job skill coverage: {weighted_coverage * 100:.0f}%."
            )
        )

    word_count = len(resume_text.split())

    if 200 <= word_count <= 1200:
        ats_score += 15
        checks.append(
            ("pass", f"Resume length looks reasonable ({word_count} words).")
        )
    elif word_count < 200:
        ats_score += 5
        checks.append(
            ("warning", f"Resume may be too short ({word_count} words).")
        )
    else:
        ats_score += 8
        checks.append(
            ("warning", f"Resume may be too long ({word_count} words).")
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
        ("Contact Information", bool(email_found or phone_found))
    )

    summary_keywords = [
        "summary",
        "professional summary",
        "profile",
        "objective",
        "career objective",
    ]
    sections.append(
        (
            "Professional Summary",
            any(keyword in text for keyword in summary_keywords),
        )
    )

    education_keywords = [
        "education",
        "university",
        "college",
        "bachelor",
        "degree",
    ]
    sections.append(
        (
            "Education",
            any(keyword in text for keyword in education_keywords),
        )
    )

    sections.append(
        (
            "Technical Skills",
            "skills" in text or len(find_skills(resume_text)) >= 3,
        )
    )

    project_keywords = [
        "project",
        "projects",
        "graduation project",
    ]
    sections.append(
        (
            "Projects",
            any(keyword in text for keyword in project_keywords),
        )
    )

    experience_keywords = [
        "work experience",
        "experience",
        "employment",
        "internship",
    ]
    sections.append(
        (
            "Work Experience",
            any(keyword in text for keyword in experience_keywords),
        )
    )

    certification_keywords = [
        "certification",
        "certifications",
        "certificate",
        "certificates",
        "course",
        "courses",
        "academy",
    ]
    sections.append(
        (
            "Certifications / Courses",
            any(keyword in text for keyword in certification_keywords),
        )
    )

    language_keywords = [
        "languages",
        "language",
        "arabic",
        "english",
    ]
    sections.append(
        (
            "Languages",
            any(keyword in text for keyword in language_keywords),
        )
    )

    return sections


# =========================================================
# RECOMMENDATIONS
# =========================================================
def generate_recommendations(
    matched_skills,
    partial_matches,
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

    if partial_matches:
        partial_text = ", ".join(
            skill.title()
            for skill in partial_matches
        )
        recommendations.append(
            f"Related experience detected for: {partial_text}. "
            "If you have direct experience with these tools, mention it explicitly."
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
    st.markdown("## 🔮 CareerLens AI")
    st.caption(
        "AI-powered career intelligence and resume analysis."
    )

    st.divider()
    st.markdown("### What CareerLens analyzes")

    st.write("🎯 Job Skill Match")
    st.write("🟡 Related Skill Match")
    st.write("🧠 Semantic Similarity")
    st.write("📊 ATS Readiness")
    st.write("📋 Resume Structure")
    st.write("💡 Smart Recommendations")

    st.divider()
    st.caption(
        "All scores are estimates and should not be treated "
        "as employer hiring decisions."
    )


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔮 CareerLens AI</div>
        <div class="hero-subtitle">
            Analyze your resume against a job description using
            semantic AI, skill matching, ATS readiness checks,
            and intelligent career recommendations.
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
        placeholder="Paste the full job description here...",
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
        st.warning("Please upload a resume PDF.")

    elif not job_description.strip():
        st.warning("Please paste a job description.")

    else:
        with st.spinner(
            "CareerLens AI is analyzing your resume..."
        ):
            try:
                resume_text = extract_text_from_pdf(uploaded_cv)
            except Exception:
                st.error("The PDF could not be read.")
                st.stop()

            if not resume_text.strip():
                st.error("No readable text was found in the PDF.")
                st.stop()

            resume_skills = find_skills(resume_text)
            job_skills = find_skills(job_description)

            matched_skills, partial_matches, missing_skills = (
                classify_skill_matches(
                    job_skills,
                    resume_skills
                )
            )

            if job_skills:
                skills_score = (
                    (
                        len(matched_skills)
                        + (len(partial_matches) * 0.5)
                    )
                    / len(job_skills)
                ) * 100
            else:
                skills_score = 0

            semantic_score = calculate_semantic_similarity(
                resume_text,
                job_description
            )

            overall_score = calculate_overall_score(
                skills_score,
                semantic_score
            )

            ats_score, ats_checks = calculate_ats_readiness(
                resume_text,
                job_skills,
                matched_skills,
                partial_matches
            )

            resume_sections = analyze_resume_sections(
                resume_text
            )

            recommendations = generate_recommendations(
                matched_skills,
                partial_matches,
                missing_skills,
                overall_score
            )

        st.success(
            "CareerLens AI analysis completed successfully."
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
        # SCORE BREAKDOWN
        # =================================================
        skills_contribution = skills_score * 0.60
        semantic_contribution = semantic_score * 0.40

        st.markdown(
            '<div class="section-title">🧮 Score Breakdown</div>',
            unsafe_allow_html=True
        )

        breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(
            3,
            gap="medium"
        )

        with breakdown_col1:
            st.markdown(
                f"""
                <div class="check-item">
                    🎯 <strong>Skills Contribution</strong><br>
                    {skills_score:.0f}% × 60% = {skills_contribution:.1f} points
                </div>
                """,
                unsafe_allow_html=True
            )

        with breakdown_col2:
            st.markdown(
                f"""
                <div class="check-item">
                    🧠 <strong>Semantic Contribution</strong><br>
                    {semantic_score:.0f}% × 40% = {semantic_contribution:.1f} points
                </div>
                """,
                unsafe_allow_html=True
            )

        with breakdown_col3:
            st.markdown(
                f"""
                <div class="check-item">
                    🏆 <strong>Overall Match</strong><br>
                    {skills_contribution:.1f} + {semantic_contribution:.1f}
                    = {overall_score:.0f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "Overall Match = 60% skill matching + 40% semantic similarity."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # MATCHED / PARTIAL / MISSING SKILLS
        # =================================================
        skill_col1, skill_col2, skill_col3 = st.columns(
            3,
            gap="large"
        )

        with skill_col1:
            st.markdown(
                '<div class="section-title">✅ Matched Skills</div>',
                unsafe_allow_html=True
            )

            if matched_skills:
                skill_html = "".join(
                    f'<span class="skill-good">✓ {skill.title()}</span>'
                    for skill in matched_skills
                )
                st.markdown(
                    skill_html,
                    unsafe_allow_html=True
                )
            else:
                st.write("No exact skill matches detected.")

        with skill_col2:
            st.markdown(
                '<div class="section-title">🟡 Related Skills</div>',
                unsafe_allow_html=True
            )

            if partial_matches:
                partial_html = "".join(
                    f'<span class="skill-partial">≈ {skill.title()}</span>'
                    for skill in partial_matches
                )
                st.markdown(
                    partial_html,
                    unsafe_allow_html=True
                )
            else:
                st.write("No related skill matches detected.")

        with skill_col3:
            st.markdown(
                '<div class="section-title">❌ Missing Skills</div>',
                unsafe_allow_html=True
            )

            if missing_skills:
                skill_html = "".join(
                    f'<span class="skill-missing">• {skill.title()}</span>'
                    for skill in missing_skills
                )
                st.markdown(
                    skill_html,
                    unsafe_allow_html=True
                )
            else:
                st.write("No missing skills detected.")

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # ATS
        # =================================================
        st.markdown(
            '<div class="section-title">📊 ATS Readiness Analysis</div>',
            unsafe_allow_html=True
        )

        st.progress(int(ats_score))

        st.caption(
            "Estimated readiness only — not an employer ATS score."
        )

        with st.expander("View detailed ATS checks"):
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
        # RESUME STRUCTURE
        # =================================================
        st.markdown(
            '<div class="section-title">📋 Resume Structure</div>',
            unsafe_allow_html=True
        )

        section_cols = st.columns(2)

        for index, (section_name, found) in enumerate(
            resume_sections
        ):
            target_col = section_cols[index % 2]

            with target_col:
                icon = "✅" if found else "⚠️"
                st.markdown(
                    f"""
                    <div class="check-item">
                        {icon} {section_name}
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
        # EXTRACTED RESUME TEXT
        # =================================================
        with st.expander("📄 View extracted resume text"):
            st.text_area(
                "Resume Text",
                resume_text,
                height=350,
                label_visibility="collapsed"
            )
