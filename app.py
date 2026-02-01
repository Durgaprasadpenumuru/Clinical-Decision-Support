import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from fpdf import FPDF
from loguru import logger

# 1. HARD-STRIP OPENAI DEPENDENCIES
# These environment variables tell the underlying LiteLLM to stop looking for OpenAI
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

load_dotenv()

# Setup Logging for Audit Trail
logger.add("triage_audit.log", rotation="500 MB", format="{time} {level} {message}")

# 2. UI CONFIG
st.set_page_config(page_title="Agentic Clinical Decision Support System", page_icon="🏥", layout="wide")

# 3. CORE LLM SETUP (Using the new LLM class for better provider isolation)
# The "groq/" prefix ensures CrewAI uses the Groq provider natively
custom_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# 4. PDF GENERATION UTILITY
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_text)
    return pdf.output(dest='S').encode('latin-1')


# 5. DASHBOARD INTERFACE
st.title("🏥 Clinical Triage & Audit System")
st.info("Verified: Running on Llama-3.3-70b-versatile")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Record Entry")
    patient_id = st.text_input("Patient ID")
    patient_note = st.text_area("Symptoms & History", height=250)
    run_button = st.button("Generate Brief", type="primary")

with col2:
    st.subheader("Physician Output")
    if run_button and patient_note:
        logger.info(f"Entry: {patient_id} | Symptoms: {patient_note}")

        with st.spinner("Executing Groq-Powered Agents..."):
            # AGENTS (Explicitly bound to custom_llm)
            analyst = Agent(
                role='Medical Scribe',
                goal='Structure clinical symptoms.',
                backstory="Expert in converting raw notes to structured medical data.",
                llm=custom_llm,
                verbose=True
            )
            triage = Agent(
                role='Triage Nurse',
                goal='Assign urgency level.',
                backstory="20-year ER veteran looking for life-threatening red flags.",
                llm=custom_llm,
                verbose=True
            )
            writer = Agent(
                role='Physician Assistant',
                goal='Write a 100-word brief for the doctor.',
                backstory="Specialized in concise medical communication.",
                llm=custom_llm,
                verbose=True
            )

            # TASKS
            t1 = Task(description=f"Parse symptoms from: {patient_note}", agent=analyst, expected_output="Symptom list")
            t2 = Task(description="Assign triage level (Red/Yellow/Green).", agent=triage,
                      expected_output="Urgency rating")
            t3 = Task(description="Provide a Markdown-formatted physician brief.", agent=writer,
                      expected_output="Physician Brief")

            # 6. THE CREW - CRITICAL SETTINGS FOR NO OPENAI
            # planning=False and embedder=None are the most important settings here
            medical_crew = Crew(
                agents=[analyst, triage, writer],
                tasks=[t1, t2, t3],
                process=Process.sequential,
                memory=False,  # Disables short/long-term memory (requires OpenAI embeddings)
                planning=False,  # Disables the Planning Agent (defaults to GPT-4o)
                embedder=None,  # Explicitly removes the default OpenAI embedder
                manager_llm=custom_llm,  # Ensures any fallback uses Groq
                verbose=True
            )

            result = medical_crew.kickoff()
            final_report = str(result)
            # Simple logic to color-code the triage level in the UI
            if "Red" in final_report or "Level 1" in final_report:
                st.error("🚨 EMERGENCY: RED PRIORITY")
            elif "Yellow" in final_report or "Level 2" in final_report:
                st.warning("⚠️ URGENT: YELLOW PRIORITY")
            else:
                st.success("✅ STABLE: GREEN PRIORITY")
            st.markdown(final_report)

            # PDF EXPORT
            pdf_data = create_pdf(final_report)
            st.download_button(
                label="📥 Download Brief (PDF)",
                data=pdf_data,
                file_name=f"Triage_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

    elif run_button and not patient_note:
        st.error("Error: Please provide symptoms to continue.")