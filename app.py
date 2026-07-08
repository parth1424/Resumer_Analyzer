import os
import uuid
from dotenv import load_dotenv

load_dotenv()
from flask import (
    Flask, request, jsonify, render_template,
    session, send_file
)

from utils.nlp_pipeline import (
    download_nltk_resources, preprocess, compute_similarity
)
from utils.pdf_extractor import (
    extract_text_from_pdf, PDFEncryptionError, PDFInvalidError
)
from utils.report_generator import build_pdf_report
from utils.skills_db import SKILLS_DB

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

with app.app_context():
    download_nltk_resources()


def match_keyword(keyword, text_lower):
    """
    Checks if keyword exists in text_lower with boundary constraints.
    Prevents false positives on short keywords (e.g., matching 'go' in 'django', 'r' in 'program').
    """
    start = 0
    while True:
        idx = text_lower.find(keyword, start)
        if idx == -1:
            return False

        # Check boundary before
        if idx > 0:
            char_before = text_lower[idx - 1]
            if char_before.isalnum():
                start = idx + 1
                continue

        # Check boundary after
        idx_after = idx + len(keyword)
        if idx_after < len(text_lower):
            char_after = text_lower[idx_after]
            if keyword[-1].isalnum() and char_after.isalnum():
                start = idx + 1
                continue

        return True


def extract_skills(text):
    text_lower = text.lower()
    matched = {}
    for category, data in SKILLS_DB.items():
        found = []
        for skill in data["keywords"]:
            if match_keyword(skill, text_lower):
                found.append(skill)
        if found:
            matched[category] = found
    return matched


def analyze_resume(resume_text, jd_text):
    cos_sim = compute_similarity(resume_text, jd_text)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    resume_flat = set()
    for skills in resume_skills.values():
        resume_flat.update(skills)

    jd_flat = set()
    for skills in jd_skills.values():
        jd_flat.update(skills)

    matching = resume_flat & jd_flat
    missing = jd_flat - resume_flat

    total_required = len(jd_flat)
    keyword_match_pct = round((len(matching) / total_required * 100) if total_required > 0 else 0, 2)

    ats_score = round((0.4 * cos_sim * 100) + (0.6 * keyword_match_pct), 2)
    ats_score = max(0.0, min(100.0, ats_score))

    strengths = []
    if len(matching) > 0:
        strengths.append(f"Resume covers {len(matching)} of {total_required} required skills")
    if cos_sim > 0.3:
        strengths.append("Good semantic alignment with the job description")
    if len(matching) >= len(jd_flat) * 0.7 and total_required > 0:
        strengths.append("Strong keyword coverage across required technologies")

    weaknesses = []
    if len(missing) > 0:
        weaknesses.append(f"Missing {len(missing)} key skills mentioned in the job description")
    if cos_sim < 0.3:
        weaknesses.append("Low semantic similarity - consider tailoring your resume language")
    if total_required == 0:
        weaknesses.append("No specific technical skills detected in job description")

    suggestions = []
    if missing:
        sugg = list(missing)
        if len(sugg) > 5:
            sugg = sugg[:5]
        suggestions.append(
            f"Add these missing skills if applicable: {', '.join(sugg)}"
        )
    if cos_sim < 0.4:
        suggestions.append(
            "Incorporate more keywords and phrases from the job description"
        )
    suggestions.append("Use bullet points with quantified achievements for impact")

    category_data = {}
    for cat in SKILLS_DB:
        jd_cat = jd_skills.get(cat, [])
        resume_cat = resume_skills.get(cat, [])
        matched_cat = [s for s in jd_cat if s in resume_cat]
        missing_cat = [s for s in jd_cat if s not in resume_cat]
        if jd_cat:
            category_data[cat] = {
                "required": len(jd_cat),
                "matched": len(matched_cat),
                "missing": len(missing_cat),
                "skills": {"matched": matched_cat, "missing": missing_cat}
            }

    return {
        "ats_score": ats_score,
        "cosine_similarity": round(cos_sim * 100, 2),
        "keyword_match_pct": keyword_match_pct,
        "total_required_skills": total_required,
        "matching_skills": sorted(matching),
        "missing_skills": sorted(missing),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "category_breakdown": category_data
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    filename = f"{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        text = extract_text_from_pdf(filepath)
    except (PDFEncryptionError, PDFInvalidError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error during text extraction: {e}"}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    session["resume_text"] = text
    return jsonify({"text": text, "length": len(text)})



@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    resume_text = data.get("resume_text") or session.get("resume_text")
    jd_text = data.get("jd_text", "")

    if not resume_text:
        return jsonify({"error": "Resume text not available. Please upload a resume first."}), 400
    if not jd_text:
        return jsonify({"error": "Job description cannot be empty"}), 400

    analysis = analyze_resume(resume_text, jd_text)
    session["analysis"] = analysis
    return jsonify(analysis)


@app.route("/report", methods=["GET"])
def report():
    analysis = session.get("analysis")
    resume_text = session.get("resume_text", "")
    jd_text = request.args.get("jd", "")

    if not analysis:
        analysis = {
            "ats_score": 0, "cosine_similarity": 0, "keyword_match_pct": 0,
            "total_required_skills": 0, "matching_skills": [],
            "missing_skills": [], "strengths": [], "weaknesses": [],
            "suggestions": [], "category_breakdown": {}
        }

    pdf_buffer = build_pdf_report(analysis, resume_text, jd_text)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="resume_analysis_report.pdf"
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
