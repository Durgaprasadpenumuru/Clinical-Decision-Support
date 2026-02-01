Agentic Clinical Decision Support System

CDS is an AI-driven medical triage and physician briefing engine. It leverages **CrewAI** and **Groq (Llama-3.3-70B)** to transform raw patient narratives into structured, actionable clinical briefs.



## 🧠 The Problem & Solution
Documentation and triage take up significant clinical bandwidth. This system uses a **Multi-Agent Systems (MAS)** approach to automate:
1. **Clinical Fact Extraction:** Converting messy patient notes into structured symptoms.
2. **Emergency Stratification:** Categorizing patients by urgency (Red/Yellow/Green).
3. **Information Compression:** Generating 100-word "Executive Briefs" for doctors.

## 🛠️ Tech Stack
- **Orchestration:** [CrewAI](https://www.crewai.com/)
- **Inference:** [Groq LPU](https://groq.com/) (Llama-3.3-70B-Versatile)
- **Framework:** [LangChain](https://www.langchain.com/)
- **Interface:** Streamlit
- **Logging:** Loguru (for immutable audit trails)

## 🏗️ Multi-Agent Architecture
The system consists of three specialized agents:
- **Medical Scribe:** Specialized in NLP for clinical entity extraction.
- **Triage Nurse:** A logic-based agent focusing on "Red Flag" detection.
- **Physician Assistant:** A synthesis agent optimized for high-density communication.

## 🚀 Installation & Setup
1. **Clone the repo:**
   ```bash
   git clone https://github.com/Durgaprasadpenumuru/Clinical-Decision-Support.git
   cd Clinical-Decision-Support
   ```
 2. **Setup Environment: Create a .env file and add your key:"**
    ```bash
    GROQ_API_KEY=your_groq_api_key_here
    ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the App:**
   ```bash
   python -m streamlit run app.py
   ```
5. **Ethical Note**
This tool is a Decision Support System (CDS) and is intended to assist, not replace, human clinical judgment.
## 👨‍🔬 Author & Maintainer

**Dr. Durga Prasad**
---
*Developed as part of an initiative to streamline Clinical Decision Support (CDS) using high-speed LPU inference.*
