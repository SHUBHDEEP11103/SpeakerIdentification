import os
import shutil
import time
import datetime
import pandas as pd
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import torch
import librosa
from speechbrain.inference.classifiers import EncoderClassifier

# --- Constants ---
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_FILE = "student_embeddings.pt"
TEST_FILE = "live_attendance.wav"
LOG_FILE = "attendance_log.csv"
DATASET_PATH = "recording2.0"
FS = 16000
SECONDS = 4
SIMILARITY_THRESHOLD = 0.50

st.set_page_config(
    page_title="JUET Voice Attendance System",
    page_icon="🎓",
    layout="wide"
)

# ── JUET THEME CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #fef9f0 !important;
    color: #2c1a0e;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 150px !important; }

/* ── Top header bar ── */
.juet-header {
    background: linear-gradient(to right, #e8720c, #8b0000);
    padding: 0;
    width: 100%;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 9999;
    border-bottom: 4px solid #d4a017;
}
.juet-header-inner {
    display: flex;
    align-items: center;
    padding: 12px 28px;
    gap: 20px;
}
.juet-logo-circle {
    width: 64px; height: 64px;
    background: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 32px;
    flex-shrink: 0;
    border: 3px solid #d4a017;
}
.juet-title-block { flex: 1; }
.juet-univ-name {
    font-size: 22px;
    font-weight: 700;
    color: white;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    margin: 0;
}
.juet-subtitle {
    font-size: 12px;
    color: #ffd580;
    letter-spacing: 0.5px;
    margin-top: 2px;
}
.juet-tagline {
    background: rgba(0,0,0,0.25);
    color: #fff3cc;
    font-size: 11px;
    text-align: center;
    padding: 4px 28px;
    border-top: 1px solid rgba(255,255,255,0.15);
    letter-spacing: 0.5px;
}

/* ── Page title strip ── */
.page-title-strip {
    background: #8b0000;
    color: white;
    padding: 8px 24px;
    font-size: 15px;
    font-weight: 600;
    margin: 0 -1rem 18px -1rem;
    border-bottom: 3px solid #e8720c;
    letter-spacing: 0.5px;
}

/* ── Navigation ── */
div[data-testid="stRadio"] > label {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6b2a0e !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stRadio"] div[role="radio"] {
    background: #fff3e0 !important;
    border: 1px solid #e8720c !important;
    border-radius: 4px !important;
    margin-bottom: 6px !important;
    padding: 8px 12px !important;
    font-size: 13px !important;
    color: #5a1a00 !important;
    transition: background 0.2s;
}
div[data-testid="stRadio"] div[role="radio"]:hover {
    background: #ffe0b2 !important;
}
div[data-testid="stRadio"] div[aria-checked="true"] {
    background: #8b0000 !important;
    color: white !important;
    border-color: #8b0000 !important;
}
div[data-testid="stRadio"] div[aria-checked="true"] p {
    color: white !important;
}

/* ── Section header ── */
h1, h2, h3 { color: #8b0000 !important; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #e8720c, #8b0000) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px;
    padding: 10px 20px !important;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }

/* ── Info / Success / Error ── */
div[data-testid="stAlert"] {
    border-radius: 4px !important;
    border-left-width: 5px !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { border: 1px solid #e0c9a6 !important; border-radius: 4px; }

/* ── Input fields ── */
input[type="text"], .stTextInput input {
    border: 1px solid #e8720c !important;
    border-radius: 4px !important;
    background: #fffdf9 !important;
}

/* ── Nav card ── */
.nav-card {
    background: linear-gradient(135deg, #8b0000, #5a0000);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    border: 1px solid #d4a017;
}
.nav-card-title {
    color: #ffd580;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.nav-card-value {
    color: white;
    font-size: 20px;
    font-weight: 700;
}

/* ── Student card ── */
.student-card {
    padding: 14px 20px;
    border-radius: 6px;
    background: white;
    border-left: 5px solid #8b0000;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* ── Section divider ── */
hr { border-color: #e8720c !important; }

/* ── Progress bar ── */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(to right, #e8720c, #8b0000) !important;
}
</style>
""", unsafe_allow_html=True)

# --- Model (cached once, stays loaded) ---
@st.cache_resource
def load_model():
    return EncoderClassifier.from_hparams(
        source=MODEL_NAME,
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

classifier = load_model()

# --- Embeddings via session_state ---
def get_embeddings():
    if "embeddings" not in st.session_state:
        refresh_embeddings()
    return st.session_state.embeddings

def refresh_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        st.session_state.embeddings = torch.load(EMBEDDING_FILE)
    else:
        st.session_state.embeddings = {}

# --- Helper Functions ---
def record_audio(filename, seconds=4):
    recording = sd.rec(int(seconds * FS), samplerate=FS, channels=1)
    sd.wait()
    write(filename, FS, recording)

def get_voiceprint(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    signal = torch.from_numpy(audio).unsqueeze(0)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze(0).squeeze(0)

def mark_attendance_in_csv(name, score):
    now = datetime.datetime.now()
    new_data = pd.DataFrame([{
        "Name": name.upper(),
        "Date": now.strftime("%Y-%m-%d"),
        "Day": now.strftime("%A"),
        "Time": now.strftime("%H:%M:%S"),
        "Confidence Score": f"{score:.2%}"
    }])
    if not os.path.exists(LOG_FILE):
        new_data.to_csv(LOG_FILE, index=False)
    else:
        new_data.to_csv(LOG_FILE, mode='a', header=False, index=False)

# ── JUET HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="juet-header">
    <div class="juet-header-inner">
        <div class="juet-logo-circle">🎓</div>
        <div class="juet-title-block">
            <div class="juet-univ-name">Jaypee University of Engineering and Technology</div>
            <div class="juet-subtitle">Smart Attendance System using Voice Recognition</div>
        </div>
    </div>
    <div class="juet-tagline">Voice-Based Smart Attendance System &nbsp;|&nbsp; Department of Computer Science & Engineering</div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ───────────────────────────────────────────────────────────────────
main_col, nav_col = st.columns([3, 1], gap="large")

with nav_col:
    student_embeddings = get_embeddings()

    # Stats cards
    st.markdown(f"""
    <div class="nav-card">
        <div class="nav-card-title">📋 Enrolled Students</div>
        <div class="nav-card-value">{len(student_embeddings)}</div>
    </div>
    """, unsafe_allow_html=True)

    now = datetime.datetime.now()
    st.markdown(f"""
    <div class="nav-card">
        <div class="nav-card-title">📅 Today</div>
        <div class="nav-card-value" style="font-size:14px;">{now.strftime("%A, %d %b %Y")}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)
    st.markdown("**🗂️ Navigation**")
    menu = ["Take Attendance", "Attendance Log", "Register New Student", "Student Directory"]
    choice = st.radio("", menu)

with main_col:
    # Page strip title
    st.markdown(f'<div class="page-title-strip">📍 {choice}</div>', unsafe_allow_html=True)

    # ── PAGE 1: Take Attendance ──────────────────────────────────────────────
    if choice == "Take Attendance":
        st.subheader("Voice-Based Attendance Marking")
        student_embeddings = get_embeddings()

        if not student_embeddings:
            st.error("⚠️ No students enrolled. Please register students first.")
        else:
            st.info(f"🔒 System active · **{len(student_embeddings)} students** enrolled · Threshold: {SIMILARITY_THRESHOLD:.0%}")

            # ── Permanent mic display area ──
            MIC_IDLE = """
            <div style="text-align:center; padding:40px 20px; background:#fff8f0;
                        border:2px solid #c8a080; border-radius:12px; margin:16px 0;">
                <style>
                    @keyframes float { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);} }
                    .mic-idle { font-size:80px; animation:float 3s ease-in-out infinite; display:inline-block; }
                </style>
                <span class="mic-idle">🎙️</span>
                <div style="font-size:18px; font-weight:600; color:#8b0000; margin-top:14px;">
                    Ready to Listen
                </div>
                <div style="font-size:13px; color:#a07050; margin-top:6px;">
                    Click the button below to start voice identification
                </div>
            </div>
            """

            MIC_ACTIVE = """
            <div style="text-align:center; padding:40px 20px; background:#fff8f0;
                        border:2px solid #8b0000; border-radius:12px; margin:16px 0;">
                <style>
                    @keyframes pulse-att  { 0%,100%{transform:scale(1);} 50%{transform:scale(1.18);} }
                    @keyframes ripple-att { 0%{opacity:.8;transform:scale(.7);} 100%{opacity:0;transform:scale(1.3);} }
                    .att-rings { position:relative; width:120px; height:120px;
                                 display:inline-flex; align-items:center; justify-content:center; }
                    .att-ring  { position:absolute; border-radius:50%; border:3px solid #8b0000;
                                 animation:ripple-att 1.5s ease-out infinite; opacity:0; }
                    .att-ring:nth-child(1){ width:76px;  height:76px;  animation-delay:0s; }
                    .att-ring:nth-child(2){ width:104px; height:104px; animation-delay:0.4s; }
                    .att-ring:nth-child(3){ width:132px; height:132px; animation-delay:0.8s; }
                    .att-mic { font-size:72px; animation:pulse-att 1s ease-in-out infinite; }
                </style>
                <div class="att-rings">
                    <div class="att-ring"></div>
                    <div class="att-ring"></div>
                    <div class="att-ring"></div>
                    <span class="att-mic">🎙️</span>
                </div>
                <div style="font-size:22px; font-weight:700; color:#8b0000; margin-top:16px;">
                    🔴 Please Speak Now...
                </div>
                <div style="font-size:13px; color:#a05a2c; margin-top:6px;">
                    Recording for 4 seconds — say your name or anything
                </div>
            </div>
            """

            mic_area = st.empty()
            mic_area.markdown(MIC_IDLE, unsafe_allow_html=True)
            result_area = st.empty()

            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            if st.button("🎤 Start Recording (4 Seconds)", use_container_width=True):
                # Animate mic
                mic_area.markdown(MIC_ACTIVE, unsafe_allow_html=True)
                result_area.empty()

                record_audio(TEST_FILE, SECONDS)

                # Return to idle while processing
                mic_area.markdown(MIC_IDLE, unsafe_allow_html=True)

                with st.spinner("Analyzing Voiceprint..."):
                    try:
                        live_embedding = get_voiceprint(TEST_FILE)
                        best_match, best_score = None, -1.0
                        for name, enrolled_emb in student_embeddings.items():
                            score = torch.nn.functional.cosine_similarity(
                                live_embedding.unsqueeze(0), enrolled_emb.unsqueeze(0)
                            ).item()
                            if score > best_score:
                                best_score = score
                                best_match = name

                        if best_score >= SIMILARITY_THRESHOLD:
                            now = datetime.datetime.now()
                            result_area.markdown(f"""
                            <div style="background:#f0fff4; border:2px solid #2c8a3e; border-radius:10px;
                                        padding:24px; text-align:center; margin-top:4px;">
                                <div style="font-size:44px;">✅</div>
                                <div style="font-size:22px; font-weight:700; color:#1a5c2e; margin-top:8px;">
                                    ATTENDANCE MARKED
                                </div>
                                <div style="font-size:30px; font-weight:700; color:#8b0000; margin:8px 0;">
                                    {best_match.upper()}
                                </div>
                                <div style="display:flex; justify-content:center; gap:40px; margin-top:12px;
                                            border-top:1px solid #c8e6c9; padding-top:12px;">
                                    <div>
                                        <div style="font-size:11px;color:#888;text-transform:uppercase;">Confidence</div>
                                        <div style="font-size:18px;font-weight:700;color:#2c8a3e;">{best_score:.2%}</div>
                                    </div>
                                    <div>
                                        <div style="font-size:11px;color:#888;text-transform:uppercase;">Time</div>
                                        <div style="font-size:18px;font-weight:700;">{now.strftime("%H:%M:%S")}</div>
                                    </div>
                                    <div>
                                        <div style="font-size:11px;color:#888;text-transform:uppercase;">Date</div>
                                        <div style="font-size:18px;font-weight:700;">{now.strftime("%d %b %Y")}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            mark_attendance_in_csv(best_match, best_score)
                        else:
                            result_area.error(f"### ❌ UNRECOGNIZED SPEAKER\n\nBest match was **{best_match}** ({best_score:.2%}) — below the threshold ({SIMILARITY_THRESHOLD:.2%}).")
                    except Exception as e:
                        result_area.error(f"Error processing audio: {e}")

    # ── PAGE 2: Attendance Log ───────────────────────────────────────────────
    elif choice == "Attendance Log":
        st.subheader("Attendance Records")
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)

            # Summary stats
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="background:#8b0000;color:white;padding:16px;border-radius:6px;text-align:center;">
                    <div style="font-size:11px;opacity:.8;">TOTAL RECORDS</div>
                    <div style="font-size:28px;font-weight:700;">{len(df)}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                unique_students = df["Name"].nunique() if "Name" in df.columns else 0
                st.markdown(f"""
                <div style="background:#e8720c;color:white;padding:16px;border-radius:6px;text-align:center;">
                    <div style="font-size:11px;opacity:.8;">UNIQUE STUDENTS</div>
                    <div style="font-size:28px;font-weight:700;">{unique_students}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                today_count = len(df[df["Date"] == today_str]) if "Date" in df.columns else 0
                st.markdown(f"""
                <div style="background:#5a0000;color:white;padding:16px;border-radius:6px;text-align:center;">
                    <div style="font-size:11px;opacity:.8;">TODAY'S ENTRIES</div>
                    <div style="font-size:28px;font-weight:700;">{today_count}</div>
                </div>""", unsafe_allow_html=True)

            st.write("")
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Download Attendance CSV", data=csv,
                               file_name='attendance_log.csv', mime='text/csv')
        else:
            st.info("No attendance records found yet.")

    # ── PAGE 3: Register New Student ─────────────────────────────────────────
    elif choice == "Register New Student":
        st.subheader("Enroll a New Student")

        st.markdown("""
        <div style="background:#fff8f0; border:1px solid #e8720c; border-radius:6px;
                    padding:14px 20px; margin-bottom:16px;">
            <strong style="color:#8b0000;">📋 Instructions</strong>
            <ul style="margin:8px 0 0 0; color:#5a1a00;">
                <li>You will record <strong>3 voice samples</strong>, each lasting <strong>5 seconds</strong>.</li>
                <li>When the 🎙️ mic appears — <strong>speak naturally</strong> (introduce yourself, count, say anything).</li>
                <li>When the ✅ appears — <strong>stop speaking</strong> and wait for the next round.</li>
                <li>A 3-second countdown is shown between each sample.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        new_name = st.text_input("Student Name (No spaces, e.g. ShubhamJha)")

        if new_name:
            person_dir = os.path.join(DATASET_PATH, new_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)

            if st.button("🎙️ Record 3 Samples & Register", type="primary"):
                progress_bar = st.progress(0)
                stage_text   = st.empty()
                guide_box    = st.empty()
                person_embs  = []

                for i in range(3):
                    stage_text.markdown(f"### Sample {i+1} of 3")
                    if i > 0:
                        for countdown in range(3, 0, -1):
                            guide_box.markdown(f"""
                            <div style="text-align:center;padding:20px;border-radius:8px;
                                        background:#fff8f0;border:1px solid #e8720c;">
                                <div style="font-size:50px;">⏳</div>
                                <div style="font-size:22px;color:#8b0000;margin-top:8px;font-weight:600;">
                                    Get ready... <strong>{countdown}</strong>
                                </div>
                            </div>""", unsafe_allow_html=True)
                            time.sleep(1)

                    guide_box.markdown(f"""
                    <div style="text-align:center;padding:28px;border-radius:8px;
                                background:#fff8f0;border:2px solid #8b0000;">
                        <style>
                            @keyframes pr {{ 0%,100%{{transform:scale(1);}} 50%{{transform:scale(1.2);}} }}
                            @keyframes rr {{ 0%{{opacity:.7;transform:scale(.7);}} 100%{{opacity:0;transform:scale(1.3);}} }}
                            .rr-rings{{position:relative;width:110px;height:110px;display:inline-flex;align-items:center;justify-content:center;}}
                            .rr-ring{{position:absolute;border-radius:50%;border:3px solid #8b0000;animation:rr 1.5s ease-out infinite;opacity:0;}}
                            .rr-ring:nth-child(1){{width:70px;height:70px;animation-delay:0s;}}
                            .rr-ring:nth-child(2){{width:95px;height:95px;animation-delay:0.4s;}}
                            .rr-ring:nth-child(3){{width:120px;height:120px;animation-delay:0.8s;}}
                            .rr-mic{{font-size:64px;animation:pr 1s ease-in-out infinite;}}
                        </style>
                        <div class="rr-rings">
                            <div class="rr-ring"></div><div class="rr-ring"></div><div class="rr-ring"></div>
                            <span class="rr-mic">🎙️</span>
                        </div>
                        <div style="font-size:22px;font-weight:700;color:#8b0000;margin-top:12px;">
                            🔴 SPEAK NOW — 5 seconds
                        </div>
                        <div style="font-size:13px;color:#a05a2c;margin-top:4px;">
                            Say anything: your name, count to 10, introduce yourself...
                        </div>
                    </div>""", unsafe_allow_html=True)

                    sample_file = os.path.join(person_dir, f"sample_{i}.wav")
                    record_audio(sample_file, 5)

                    guide_box.markdown(f"""
                    <div style="text-align:center;padding:24px;border-radius:8px;
                                background:#f0fff4;border:2px solid #2c8a3e;">
                        <div style="font-size:48px;">✅</div>
                        <div style="font-size:20px;font-weight:700;color:#2c8a3e;margin-top:10px;">
                            STOP — Sample {i+1} captured!
                        </div>
                        <div style="font-size:13px;color:#555;margin-top:4px;">Processing...</div>
                    </div>""", unsafe_allow_html=True)

                    emb = get_voiceprint(sample_file)
                    person_embs.append(emb)
                    progress_bar.progress((i + 1) / 3)
                    time.sleep(1)

                guide_box.empty()
                stage_text.empty()

                avg_emb = torch.mean(torch.stack(person_embs), dim=0)
                embs = get_embeddings()
                embs[new_name] = avg_emb
                torch.save(embs, EMBEDDING_FILE)
                refresh_embeddings()

                st.success(f"🎉 Successfully registered **{new_name}**! They can now mark attendance.")
                st.balloons()

    # ── PAGE 4: Student Directory ────────────────────────────────────────────
    elif choice == "Student Directory":
        st.subheader("Enrolled Student Directory")

        if "to_remove" not in st.session_state:
            st.session_state.to_remove = None

        student_embeddings = get_embeddings()

        if not student_embeddings:
            st.info("No students enrolled yet. Use 'Register New Student' to add students.")
        else:
            st.write(f"**{len(student_embeddings)} student(s) currently enrolled.**")
            st.markdown("<hr>", unsafe_allow_html=True)

            for student_name in list(student_embeddings.keys()):
                person_dir = os.path.join(DATASET_PATH, student_name)
                file_count = len([f for f in os.listdir(person_dir) if f.endswith(".wav")]) \
                             if os.path.isdir(person_dir) else 0

                card_col, btn_col = st.columns([5, 1])
                with card_col:
                    st.markdown(f"""
                    <div class="student-card">
                        <span style="font-size:18px;">🎓</span>
                        <strong style="font-size:16px; margin-left:10px; color:#5a0000;">{student_name}</strong>
                        <span style="font-size:12px; color:#999; margin-left:16px;">
                            {file_count} voice sample(s) on record
                        </span>
                    </div>""", unsafe_allow_html=True)
                with btn_col:
                    st.write("")
                    if st.button("🗑️ Remove", key=f"remove_{student_name}"):
                        st.session_state.to_remove = student_name
                        st.rerun()

            if st.session_state.to_remove:
                pending = st.session_state.to_remove
                st.warning(f"⚠️ Are you sure you want to remove **{pending}**? This will permanently delete all their voice data.")
                yes_btn, no_btn, _ = st.columns([1, 1, 4])
                with yes_btn:
                    if st.button("✅ Yes, Remove", key="confirm_remove"):
                        embs = get_embeddings()
                        if pending in embs:
                            del embs[pending]
                            torch.save(embs, EMBEDDING_FILE)
                        person_dir = os.path.join(DATASET_PATH, pending)
                        if os.path.isdir(person_dir):
                            shutil.rmtree(person_dir)
                        st.session_state.to_remove = None
                        refresh_embeddings()
                        st.rerun()
                with no_btn:
                    if st.button("❌ Cancel", key="cancel_remove"):
                        st.session_state.to_remove = None
                        st.rerun()

# ── FOOTER (fixed at bottom) ────────────────────────────────────────────────
st.markdown("""
<style>
.juet-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 10px 28px;
    background: linear-gradient(to right, #e8720c, #8b0000);
    color: white;
    text-align: center;
    font-size: 12px;
    border-top: 3px solid #d4a017;
    z-index: 9999;
}
</style>
<div class="juet-footer">
    © Jaypee University of Engineering and Technology &nbsp;|&nbsp;
    Smart Attendance System using Voice Recognition &nbsp;|&nbsp; B.Tech CSE Minor Project
</div>
""", unsafe_allow_html=True)
