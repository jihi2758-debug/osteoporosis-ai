# app.py - OA Premium Dashboard (Auth) + Inference Logs + Save-after-detection (Option C)
import streamlit as st
from streamlit_option_menu import option_menu
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_chat import message
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import os
from fpdf import FPDF
from werkzeug.security import generate_password_hash, check_password_hash
import io
from datetime import datetime

# -----------------------
# Config
# -----------------------
DB_PATH = "database.db"
MODEL_PATH = "models/E6_albumentations.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

st.set_page_config(page_title="OA Premium Dashboard (Auth + Logs)", layout="wide", page_icon="🦴")

# Load CSS (keep your premium CSS in assets/style.css)
if os.path.exists("assets/style.css"):
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------
# Database helpers & schema (adds inference_logs)
# -----------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    # users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT
    );
    """)
    # patients table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE,
        name TEXT,
        age INTEGER,
        gender TEXT,
        last_visit TEXT,
        notes TEXT,
        created_by INTEGER,
        FOREIGN KEY (created_by) REFERENCES users(id)
    );
    """)
    # inference logs table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS inference_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        predicted_grade TEXT,
        timestamp TEXT,
        user_id INTEGER,
        orig_image_path TEXT,
        heatmap_path TEXT,
        notes TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """)
    conn.commit()
    conn.close()

# initialize DB on startup (creates tables if absent)
init_db()

# -----------------------
# Authentication helpers
# -----------------------
def register_user(username, password, full_name=""):
    conn = get_connection()
    cur = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        cur.execute("INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)", (username, pw_hash, full_name))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, "Username already exists"
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row:
        if check_password_hash(row["password_hash"], password):
            return {"id": row["id"], "username": row["username"], "full_name": row["full_name"]}
    return None

def get_user_by_username(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row

# -----------------------
# Patient DB operations
# -----------------------
def create_patient(patient_id, name, age, gender, last_visit, notes, created_by):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO patients (patient_id, name, age, gender, last_visit, notes, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, name, age, gender, last_visit, notes, created_by))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, "Patient ID already exists"
    finally:
        conn.close()

def read_patients(limit=200):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT p.*, u.username AS created_by_username FROM patients p LEFT JOIN users u ON p.created_by = u.id ORDER BY p.id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_patient(row_id, **fields):
    conn = get_connection()
    cur = conn.cursor()
    keys = ", ".join([f"{k} = ?" for k in fields.keys()])
    vals = list(fields.values()) + [row_id]
    cur.execute(f"UPDATE patients SET {keys} WHERE id = ?", vals)
    conn.commit()
    conn.close()

def delete_patient(row_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM patients WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

def get_patient_by_patient_id(patient_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

# -----------------------
# Inference log helpers
# -----------------------
def log_inference(patient_id, predicted_grade, user_id, orig_image_path, heatmap_path, notes=""):
    conn = get_connection()
    cur = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO inference_logs (patient_id, predicted_grade, timestamp, user_id, orig_image_path, heatmap_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, predicted_grade, timestamp, user_id, orig_image_path, heatmap_path, notes))
    conn.commit()
    conn.close()

def read_inference_logs(limit=50):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT l.*, u.username AS run_by, p.name AS patient_name
        FROM inference_logs l
        LEFT JOIN users u ON l.user_id = u.id
        LEFT JOIN patients p ON l.patient_id = p.patient_id
        ORDER BY l.id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# -----------------------
# Model loading (keeps previous behaviour)
# -----------------------
@st.cache_resource
def load_model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=False)
    model.fc = nn.Linear(2048, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

MODEL_AVAILABLE = os.path.exists(MODEL_PATH)
model = None
if MODEL_AVAILABLE:
    try:
        model = load_model()
    except Exception as e:
        st.warning("Model load failed: " + str(e))
        MODEL_AVAILABLE = False

transform = T.Compose([T.Resize((224,224)), T.ToTensor()])

def generate_gradcam(model, x):
    # robust last conv selection
    try:
        last_conv = model.layer4[2].conv3
    except Exception:
        last_conv = None
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                last_conv = m
                break
    act, grad = None, None
    def fwd(m, i, o): 
        nonlocal act; act = o
    def bwd(m, gi, go):
        nonlocal grad; grad = go[0]
    handle_fwd = last_conv.register_forward_hook(fwd)
    handle_bwd = last_conv.register_backward_hook(bwd)
    out = model(x)
    cls = out.argmax().item()
    model.zero_grad()
    out[0, cls].backward()
    cam = (grad.mean(dim=[2,3], keepdim=True) * act).sum(1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = np.uint8(cam * 255)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    handle_fwd.remove()
    handle_bwd.remove()
    return heatmap, cls

def generate_pdf_report(pred, original_path, heatmap_path, patient_info=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="AI Osteoarthritis Report", ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", size=12)
    if patient_info:
        pdf.cell(0, 8, txt=f"Patient: {patient_info.get('name','-')}  |  ID: {patient_info.get('patient_id','-')}", ln=True)
        pdf.ln(4)
    pdf.cell(0, 8, txt=f"Predicted Grade: {pred}", ln=True)
    pdf.ln(8)
    pdf.image(original_path, x=15, y=60, w=80)
    pdf.image(heatmap_path, x=110, y=60, w=80)
    out = "tmp/report.pdf"
    os.makedirs("tmp", exist_ok=True)
    pdf.output(out)
    return out

# -----------------------
# Session-state for auth & chat
# -----------------------
if "user" not in st.session_state:
    st.session_state["user"] = None  # will hold dict of user when logged in

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -----------------------
# Sidebar: Auth + Navigation
# -----------------------
with st.sidebar:
    st.markdown("<div class='glass-card'><h3 style='margin:0;'>🏥 OA Premium</h3><div class='small-muted'>Glass UI • Auth • Patient DB</div></div>", unsafe_allow_html=True)
    st.markdown("---")

    # If not logged in -> show login/register forms
    if st.session_state["user"] is None:
        # Simple tabs for login/register
        auth_tab = st.radio("Auth:", ["Login", "Register"])
        if auth_tab == "Register":
            st.write("### Create account")
            new_username = st.text_input("Username", key="reg_user")
            new_fullname = st.text_input("Full name", key="reg_name")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Register", key="reg_btn"):
                if not new_username or not new_password:
                    st.error("Username & password required")
                else:
                    ok, err = register_user(new_username.strip(), new_password.strip(), new_fullname.strip())
                    if ok:
                        st.success("Account created — please log in")
                    else:
                        st.error(err)
        else:
            st.write("### Login")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", key="login_btn"):
                user = authenticate_user(username.strip(), password.strip())
                if user:
                    st.session_state["user"] = user
                    st.success(f"Welcome, {user.get('username')}")
                else:
                    st.error("Invalid username or password")
        st.markdown("---")
        st.markdown("<div class='small-muted'>You must be logged in to manage patients & run detections.</div>", unsafe_allow_html=True)
    else:
        # Logged-in view: show nav menu and logout
        st.markdown(f"<div class='small-muted'>Logged in as <b>{st.session_state['user']['username']}</b></div>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state["user"] = None
            st.experimental_rerun()

    st.markdown("---")
    # Navigation (available to everyone, but patient pages will enforce login)
    choice = option_menu(None,
                         ["Dashboard","Patients","AI Detector","Analytics","Chat Assistant","Settings","About"],
                         icons=['house','people','activity','bar-chart','chat','gear','info-circle'],
                         menu_icon="cast", default_index=0)

# -----------------------
# Header
# -----------------------
st.markdown("<div class='header-wrap'><div class='header-glow'></div><div class='h1-glass'><img src='https://img.icons8.com/ios-filled/30/2fb7b7/knee.png'/> <div><div style='font-size:20px;font-weight:700'>OA Premium Dashboard</div><div class='small-muted'>Glassmorphism • Secure • Inference Logs</div></div></div></div>", unsafe_allow_html=True)

# -----------------------
# Pages with Auth enforcement for patient pages
# -----------------------
if choice == "Dashboard":
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Overview")
        k1, k2, k3, k4 = st.columns(4)
        # use real counts from DB (best-effort)
        logs = read_inference_logs(limit=1000)
        total_processed = len(logs)
        avg_grade = "—"
        if logs:
            # map grade names to numeric (Grade 0 -> 0)
            nums = []
            for l in logs:
                try:
                    nums.append(int(l["predicted_grade"].split()[-1]))
                except Exception:
                    pass
            if nums:
                avg_grade = round(sum(nums)/len(nums), 2)
        k1.metric("X-rays processed", total_processed)
        k2.metric("Avg. Grade", avg_grade)
        k3.metric("Today", sum(1 for l in logs if l["timestamp"].startswith(datetime.utcnow().date().isoformat())), delta=None)
        k4.metric("Model GPU", torch.cuda.is_available())
        st.markdown("### Recent Predictions")
        recent_logs = read_inference_logs(limit=10)
        if recent_logs:
            df_logs = pd.DataFrame(recent_logs)
            # show useful columns
            display_cols = df_logs[["id","patient_id","patient_name","predicted_grade","timestamp","run_by","notes"]]
            st.dataframe(display_cols, use_container_width=True)
        else:
            st.info("No inference logs yet.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Grade distribution (from logs)")
        # build distribution from logs if present otherwise demo
        if logs:
            cnts = {}
            for l in logs:
                g = l["predicted_grade"]
                cnts[g] = cnts.get(g, 0) + 1
            dist_df = pd.DataFrame({"grade": list(cnts.keys()), "count": list(cnts.values())})
        else:
            dist_df = pd.DataFrame({"grade":[0,1,2,3,4], "count":[80,200,340,420,284]})
        fig = px.pie(dist_df, names="grade", values="count", title="Grade Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif choice == "Patients":
    st.subheader("Patient Registry")
    if st.session_state["user"] is None:
        st.warning("You must be logged in to view and manage patients. Please login (sidebar).")
    else:
        # CRUD UI
        patients = read_patients(limit=1000)
        df = pd.DataFrame(patients)
        if df.empty:
            st.info("No patients found. Add one below.")
        else:
            st.dataframe(df[["id","patient_id","name","age","gender","last_visit","created_by_username","notes"]], use_container_width=True)

        st.markdown("### Add / Edit Patient")
        with st.form("patient_form"):
            pid = st.text_input("Patient ID")
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=40)
            gender = st.selectbox("Gender", ["M","F","Other"])
            last_visit = st.date_input("Last visit")
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Add Patient")
            if submitted:
                ok, err = create_patient(pid.strip(), name.strip(), int(age), gender, last_visit.strftime("%Y-%m-%d"), notes.strip(), st.session_state["user"]["id"])
                if ok:
                    st.success("Patient created")
                else:
                    st.error(err)

        st.markdown("### Manage existing patients")
        sel = st.selectbox("Pick patient to edit/delete", options=[(p["id"], f'{p["patient_id"]} - {p["name"]}') for p in patients], format_func=lambda x: x[1]) if patients else None
        if sel:
            sel_id = sel[0]
            rec = next((p for p in patients if p["id"] == sel_id), None)
            if rec:
                c1, c2 = st.columns(2)
                with c1:
                    new_name = st.text_input("Edit Name", value=rec["name"])
                    new_age = st.number_input("Edit Age", value=rec["age"])
                    new_gender = st.selectbox("Edit Gender", ["M","F","Other"], index=["M","F","Other"].index(rec["gender"]) if rec["gender"] in ["M","F","Other"] else 0)
                with c2:
                    new_last = st.date_input("Edit Last Visit", value=pd.to_datetime(rec["last_visit"]))
                    new_notes = st.text_area("Edit Notes", value=rec["notes"])
                if st.button("Update Patient"):
                    update_patient(sel_id, name=new_name, age=new_age, gender=new_gender, last_visit=new_last.strftime("%Y-%m-%d"), notes=new_notes)
                    st.success("Updated")
                if st.button("Delete Patient"):
                    delete_patient(sel_id)
                    st.success("Deleted")

elif choice == "AI Detector":
    st.subheader("X-ray AI Detector (Grad-CAM)")
    if st.session_state["user"] is None:
        st.warning("You must be logged in to run detections.")
    uploaded = st.file_uploader("Upload a Knee X-ray", type=["jpg","jpeg","png"])
    # we'll present flexible post-detection saving options (Option C)
    if uploaded:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded X-ray", width=380)
        x = transform(img).unsqueeze(0).to(DEVICE)
        if MODEL_AVAILABLE and model is not None:
            with st.spinner("Analyzing image..."):
                try:
                    heatmap, cls = generate_gradcam(model, x)
                    grade = CLASSES[cls]
                    st.success(f"Predicted: {grade}")
                    img_np = np.array(img.resize((224,224)))
                    overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)
                    st.image(overlay, caption="Grad-CAM Overlay", use_column_width=False)
                    # save images (unique filenames per run)
                    os.makedirs("tmp", exist_ok=True)
                    timestamp_short = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    orig_path = f"tmp/orig_{timestamp_short}.jpg"
                    heat_path = f"tmp/heat_{timestamp_short}.jpg"
                    img.save(orig_path)
                    cv2.imwrite(heat_path, overlay)

                    # --- Option C UI: ask user what to do next ---
                    st.markdown("### Save prediction?")
                    save_choice = st.radio("Choose how to save this prediction (Option C):",
                                           ("Save to existing patient", "Create new patient & save", "Don't save"))
                    if save_choice == "Save to existing patient":
                        # list patients by patient_id - name
                        patients = read_patients(limit=1000)
                        if not patients:
                            st.info("No patients available — create one or select 'Create new patient & save'.")
                        else:
                            options = [ (p["patient_id"], f'{p["patient_id"]} - {p["name"]}') for p in patients ]
                            sel = st.selectbox("Select patient", options=options, format_func=lambda x: x[1])
                            if st.button("Attach prediction to selected patient"):
                                selected_patient_id = sel[0]
                                # log inference
                                log_inference(selected_patient_id, grade, st.session_state["user"]["id"], orig_path, heat_path, notes="")
                                # update patient's last_visit and append notes
                                patient = get_patient_by_patient_id(selected_patient_id)
                                new_notes = (patient.get("notes","") or "") + f"\nInference on {datetime.utcnow().isoformat()}: {grade}"
                                update_patient(patient["id"], last_visit=datetime.utcnow().strftime("%Y-%m-%d"), notes=new_notes)
                                st.success(f"Saved prediction to patient {selected_patient_id}")
                    elif save_choice == "Create new patient & save":
                        with st.form("create_patient_and_save"):
                            cp_patient_id = st.text_input("New Patient ID")
                            cp_name = st.text_input("Name")
                            cp_age = st.number_input("Age", min_value=0, max_value=120, value=40)
                            cp_gender = st.selectbox("Gender", ["M","F","Other"])
                            cp_last_visit = st.date_input("Last visit")
                            cp_notes = st.text_area("Notes (optional)")
                            cp_submit = st.form_submit_button("Create patient & save prediction")
                            if cp_submit:
                                if not cp_patient_id or not cp_name:
                                    st.error("Patient ID and Name required")
                                else:
                                    ok, err = create_patient(cp_patient_id.strip(), cp_name.strip(), int(cp_age), cp_gender, cp_last_visit.strftime("%Y-%m-%d"), cp_notes.strip(), st.session_state["user"]["id"])
                                    if ok:
                                        # log inference with this patient id
                                        log_inference(cp_patient_id.strip(), grade, st.session_state["user"]["id"], orig_path, heat_path, notes=cp_notes.strip())
                                        st.success(f"Patient {cp_patient_id} created and prediction saved.")
                                    else:
                                        st.error(err)
                    else:  # Don't save
                        st.info("Prediction not saved. You can still Download PDF or change choice.")
                        # allow PDF download anyway
                        if st.button("Download PDF Report (unsaved)"):
                            patient_info = {"patient_id":"Unassigned", "name":""}
                            pdf_path = generate_pdf_report(grade, orig_path, heat_path, patient_info=patient_info)
                            with open(pdf_path, "rb") as f:
                                st.download_button("⬇ Download Report (Unassigned)", f, file_name="OA_report_unassigned.pdf")
                    # also show a direct PDF download button for convenience
                    if st.button("Download PDF Report (saved/unsaved)"):
                        patient_info = {"patient_id":"", "name":""}
                        pdf_path = generate_pdf_report(grade, orig_path, heat_path, patient_info=patient_info)
                        with open(pdf_path, "rb") as f:
                            st.download_button("⬇ Download Report", f, file_name=f"OA_report_{timestamp_short}.pdf")
                except Exception as e:
                    st.error("Grad-CAM failed: " + str(e))
        else:
            st.warning("Model not available — cannot run prediction. You can still view the image.")
        st.markdown("</div>", unsafe_allow_html=True)

elif choice == "Analytics":
    st.subheader("Advanced Analytics")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    counts = np.random.poisson(20, size=30).cumsum()
    df_ts = pd.DataFrame({"date": dates, "processed": counts})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ts["date"], y=df_ts["processed"], mode="lines", name="Processed"))
    fig.update_layout(title="X-rays processed over time", xaxis_title="Date", yaxis_title="Cumulative")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### KPI Correlations (demo)")
    df_corr = pd.DataFrame(np.random.rand(6,6), columns=list("ABCDEF"))
    fig2 = px.imshow(df_corr, text_auto=True, aspect="auto", title="Correlation matrix (demo)")
    st.plotly_chart(fig2, use_container_width=True)

elif choice == "Chat Assistant":
    st.subheader("AI Chat Assistant (local)")
    user_msg = st.text_input("Ask the assistant about OA, model, or workflows:")
    if st.button("Send"):
        if user_msg.strip():
            st.session_state["chat_history"].append({"role":"user","text":user_msg})
            reply = f"Assistant: Received: '{user_msg}'. You are logged in as: {st.session_state['user']['username'] if st.session_state['user'] else 'Guest'}"
            st.session_state["chat_history"].append({"role":"bot","text":reply})
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for m in st.session_state["chat_history"]:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['text']}")
        else:
            st.markdown(f"**Bot:** {m['text']}")
    st.markdown("</div>", unsafe_allow_html=True)

elif choice == "Settings":
    st.subheader("Settings")
    st.write("Small preferences")
    st.checkbox("Enable debug logs", value=False)
    st.checkbox("Show advanced model info", value=False)

elif choice == "About":
    st.subheader("About")
    st.markdown("""
    **OA Premium Dashboard — Auth + Inference Logs**  
    - SQLite-backed users & patients  
    - Inference logs (who ran what & when)  
    - Option C workflow: attach predictions to existing patient / create new / don't save  
    - Passwords hashed (Werkzeug)
    """)
    st.markdown("**Developer:** Yash Singh")

# Footer
st.markdown("<div style='margin-top:18px; color: rgba(255,255,255,0.45); font-size:12px;'>Built with ❤️ — Glassmorphism Premium • Auth • Inference Logs • v1.2</div>", unsafe_allow_html=True)
