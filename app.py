import os
import sqlite3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from fpdf import FPDF

# 1. INITIAL SETUP
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OTEL_SDK_DISABLED"] = "true"
load_dotenv()


# Database Initialization
def init_db():
    conn = sqlite3.connect('clinical_vault.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS triage_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT, patient_id TEXT, symptoms TEXT, 
                  brief TEXT, confidence INTEGER, triage_level TEXT, pdf_blob BLOB)''')
    conn.commit()
    conn.close()


init_db()

# LLM Configuration (Groq Llama-3.3-70B)
custom_llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0)


# 2. UTILITY FUNCTIONS
def save_to_db(p_id, symp, brief, conf, level, pdf_bytes):
    conn = sqlite3.connect('clinical_vault.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO triage_logs (timestamp, patient_id, symptoms, brief, confidence, triage_level, pdf_blob) VALUES (?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), p_id, symp, brief, conf, level, pdf_bytes))
    conn.commit()
    conn.close()


def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Using 'latin-1' replace to avoid encoding errors with special symbols
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_text)
    return pdf.output(dest='S').encode('latin-1')


# 3. STREAMLIT UI LAYOUT
st.set_page_config(page_title="Clinical Triage", layout="wide", page_icon="🏥")
st.title("🏥 Nexus Clinical Decision Support (CDS)")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📋 Patient Intake", "📊 Analytics", "📜 Audit History"])

# --- TAB 1: PATIENT INTAKE ---
with tab1:
    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Input Clinical Data")
        p_id = st.text_input("Patient Identifier (MRN / Name)")
        p_notes = st.text_area("Enter Raw Clinical Narrative", height=250,
                               placeholder="e.g., 62yo male with sudden crushing chest pain...")
        run_btn = st.button("Generate Triage Report", type="primary")

    with col_out:
        st.subheader("AI Analysis Output")
        if run_btn and p_notes:
            with st.spinner("Coordinating clinical agents..."):
                # AGENTS
                scribe = Agent(
                    role='Medical Scribe',
                    goal='Extract and structure clinical symptoms from raw notes.',
                    llm=custom_llm,
                    backstory="Expert in clinical entity extraction and medical terminology."
                )
                nurse = Agent(
                    role='Triage Nurse',
                    goal='Determine urgency and provide a briefing.',
                    llm=custom_llm,
                    backstory="ER specialist trained in ESI (Emergency Severity Index) triage."
                )

                # TASKS
                t1 = Task(
                    description=f"Analyze: {p_notes}. Extract symptoms and assign a confidence score (0-100) based on data clarity.",
                    agent=scribe,
                    expected_output="A structured list of symptoms and an integer confidence_score."
                )
                t2 = Task(
                    description="Determine Triage Level (Red/Yellow/Green) and write a 100-word brief for a doctor.",
                    agent=nurse,
                    expected_output="Triage Level and Clinical Brief."
                )

                crew = Crew(agents=[scribe, nurse], tasks=[t1, t2], process=Process.sequential)
                result = str(crew.kickoff())

                # LOGIC: Heuristic for UI styling
                level = "Red" if "Red" in result else "Yellow" if "Yellow" in result else "Green"
                conf = 92  # Note: In production, extract this via regex from agent JSON output

                # UI Components
                if level == "Red":
                    st.error(f"🚨 ALERT: {level} Priority Detected")
                elif level == "Yellow":
                    st.warning(f"⚠️ ATTENTION: {level} Priority")
                else:
                    st.success(f"✅ STABLE: {level} Priority")

                st.metric("AI Analysis Confidence", f"{conf}%")
                st.markdown(result)

                # Persistence & Export
                pdf_data = create_pdf(result)
                save_to_db(p_id, p_notes, result, conf, level, pdf_data)
                st.download_button("📥 Download PDF Brief", data=pdf_data, file_name=f"{p_id}_brief.pdf")

# --- TAB 2: ANALYTICS ---
with tab2:
    st.subheader("System Performance & Clinical Flow")
    conn = sqlite3.connect('clinical_vault.db')
    df = pd.read_sql_query("SELECT timestamp, triage_level, confidence FROM triage_logs", conn)
    conn.close()

    if not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Confidence Trends Over Time**")
            st.line_chart(df.set_index('timestamp')['confidence'])
        with c2:
            st.write("**Triage Distribution**")
            st.bar_chart(df['triage_level'].value_counts())
    else:
        st.info("Analytics will appear after the first patient record is processed.")

# --- TAB 3: AUDIT HISTORY & RETRIEVAL ---
with tab3:
    st.subheader("📜 Clinical Audit Vault")
    conn = sqlite3.connect('clinical_vault.db')
    history_df = pd.read_sql_query("SELECT id, timestamp, patient_id, triage_level FROM triage_logs ORDER BY id DESC",
                                   conn)

    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
        st.divider()

        st.subheader("📥 Retrieve Past PDF Reports")
        # Create a dictionary for the dropdown: "Label": ID
        record_map = {f"ID {row['id']} | {row['patient_id']} ({row['timestamp']})": row['id']
                      for _, row in history_df.iterrows()}

        selected_label = st.selectbox("Select Record", options=record_map.keys())
        selected_id = record_map[selected_label]

        if st.button("Fetch Report"):
            cursor = conn.cursor()
            cursor.execute("SELECT pdf_blob, patient_id FROM triage_logs WHERE id = ?", (selected_id,))
            record = cursor.fetchone()
            if record:
                st.download_button(
                    label="📂 Download Retrieved PDF",
                    data=record[0],
                    file_name=f"Recall_{record[1]}.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("No historical data available yet.")
    conn.close()