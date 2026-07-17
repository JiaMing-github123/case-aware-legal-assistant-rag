import os
import re
import glob
print("1. launching")
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
print("1. import done")
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
print("importing sentence-transformers...")
from sentence_transformers import CrossEncoder
import time
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv

# 0. Initialise Evaluation Metrics
def init_metrics_db():
    conn = sqlite3.connect("metrics.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS rag_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT,
            latency REAL,
            truthfulness_score REAL,
            completeness_score REAL,
            consistency_score REAL,
            context_relevance_score REAL, 
            context_sufficiency_score REAL,  
            error_category TEXT,
            error_reason TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_metrics_db()

# 1. FastAPI 
app = FastAPI(title="LexAI Backend API")
print("4. fastAPI done")

@app.get("/")
async def serve_frontend():
    # Ensure legal-assistant.html and api_server.py in same folder
    return FileResponse("ui/legal-assistant.html")


# Allow front-end HTML (even when opened locally via `file://`) to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Load environment variables from the .env file
load_dotenv()

# Retrieve the API key securely from the environment variables
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("Cant find GROQ_API_KEY in environment variables. Please set it in the .env file.")

os.environ["GROQ_API_KEY"] = groq_key

llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
print("2. sentence-transformers...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("3. sentence-transformers...done")
retriever_act = None
retriever_cases = None

# Initialising LLM Judge
judge_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0).bind(
    response_format={"type": "json_object"}
)

print("🚀 Starting Server and Initializing Knowledge Base...")

# Loading Act 177
act_documents = []
pdf_file_path = "data/act_177.pdf"
if os.path.exists(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    raw_pages = loader.load()
    full_text = "\n".join([page.page_content for page in raw_pages])
    full_text = re.sub(
        r"Laws of Malaysia|Act 177|Industrial Relations|^\s*\d+\s*$",
        "",
        full_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    start_match = re.search(
        r"An Act to promote and maintain industrial harmony", full_text
    )
    if start_match:
        full_text = full_text[start_match.start() :]

    schedules_split = re.split(r"(?i)\n(?=FIRST\s+SCHEDULE\s*\n|SECOND\s+SCHEDULE\s*\n)", full_text)
    print(f"Check PDF splitting: Split {len(schedules_split)} large chunks (if there is only one chunk, it means the splitting has failed!)")
    main_act_text = schedules_split[0]

    chunks = re.split(r"(?m)^(\d+[A-Za-z]?\.)\s", main_act_text)
    act_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", r"(?=\(\d+\))", r"(?=\([a-z]\))", ".", " "],
    )

    current_part = "PRELIMINARY"
    for i in range(1, len(chunks), 2):
        sec_num, sec_content = chunks[i].strip(), chunks[i + 1].strip()
        prev_chunk_lines = [
            line.strip() for line in chunks[i - 1].split("\n") if line.strip()
        ]
        sec_title = prev_chunk_lines[-1] if prev_chunk_lines else "Unknown"
        part_match = re.search(r"(PART\s+[IXV]+.*?)(?=\n)", chunks[i - 1])
        if part_match:
            current_part = part_match.group(1).strip()

        full_section_header = f"Act 177 - Section {sec_num} ({sec_title})"
        combined_text = f"[{current_part}]\n{full_section_header}\n{sec_content}"

        for sub_chunk in act_text_splitter.split_text(combined_text):
            if len(sub_chunk.strip()) > 30:
                act_documents.append(
                    Document(
                        page_content=sub_chunk.strip(),
                        metadata={"source": full_section_header},
                    )
                )

    if len(schedules_split) > 1:
        for i in range(1, len(schedules_split)):
            schedule_text = schedules_split[i].strip()
            if not schedule_text:
                continue

            # Identify which schedule
            schedule_match = re.search(r"^(FIRST\s+SCHEDULE|SECOND\s+SCHEDULE)", schedule_text, re.IGNORECASE)
            schedule_title = (
                schedule_match.group(1).upper() if schedule_match else f"SCHEDULE {i}"
            )

            full_schedule_header = f"Act 177 - {schedule_title}"

            # Use the same RecursiveCharacterTextSplitter to split the schedule
            for sub_chunk in act_text_splitter.split_text(schedule_text):
                if len(sub_chunk.strip()) > 30:
                    act_documents.append(
                        Document(
                            page_content=sub_chunk.strip(),
                            metadata={
                                "source": full_schedule_header
                            },  # Ensure the metadata is correct
                        )
                    )

# Loading Cases
case_documents = []
cases_folder_path = "Cases/Docx"
if os.path.exists(cases_folder_path):
    docx_files = glob.glob(os.path.join(cases_folder_path, "*.docx"))
    case_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n\\[\\d+\\]", "\n", ".", " "],
    )
    for file_path in docx_files:
        loader = Docx2txtLoader(file_path)
        full_text = "\n".join([page.page_content for page in loader.load()])
        case_title = (
            full_text.split("\n")[0].strip()
            if len(full_text.split("\n")[0].strip()) > 10
            else os.path.basename(file_path)
        )

        for noise in [
            "Industrial Law Journal (ILJ)",
            "Industrial Law Journal Unreported (ILJU)",
            "INDUSTRIAL COURT (KUALA LUMPUR)",
            "End of Document",
        ]:
            full_text = full_text.replace(noise, "")

        for chunk_text in case_text_splitter.split_text(full_text):
            if len(chunk_text.strip()) > 50:
                case_documents.append(
                    Document(
                        page_content=chunk_text.strip(), metadata={"source": case_title}
                    )
                )

# Building Chroma
db_act = Chroma.from_documents(act_documents, embeddings, collection_name="act_177")
db_cases = Chroma.from_documents(
    case_documents, embeddings, collection_name="past_cases"
)
retriever_act = db_act.as_retriever(search_kwargs={"k": 10})
retriever_cases = db_cases.as_retriever(search_kwargs={"k": 10})

print(
    f" Knowledge Base Ready! Act 177: {len(act_documents)} chunks, Cases: {len(case_documents)} chunks."
)

# 3. Chains Prompt Define
refine_template = """You are an expert legal search assistant tailored for Malaysian Industrial Relations Law (Act 177). 
Extract the core legal issue from the user's query into a SINGLE, concise semantic phrase (5 to 15 words). 
RULES:
1. Output ONLY the short phrase. No quotes, no explanations, no introductory text.
2. OMIT grammatical filler words.
3. STRICT NOUN PRESERVATION: NEVER translate specific diseases (e.g., Covid, COVID-19) into general terms like 'illness'. MUST RETAIN the exact specific noun.
4. TRANSLATION & CONTEXT CRITICAL ANCHORING: 
   - If the user implies being "fired", "sacked", or "terminated" -> use "dismissal without just cause section 20".
   - If the user implies being "forced to resign" or "quit because of bad treatment" -> use "constructive dismissal section 20 dismissal without just cause".
   - CRITICAL RETAIN: Append specific factual keywords from the query (e.g., "pregnant", "covid", "late").

EXAMPLES:
- "I was fired because I am pregnant and afraid of Covid" -> dismissal without just cause pregnancy covid section 20
- "My boss made my life miserable so I had to quit" -> constructive dismissal section 20 dismissal without just cause miserable treatment
Original: {original_query}
Core Legal Phrase:"""
refine_chain = (
    ChatPromptTemplate.from_template(refine_template) | llm | StrOutputParser()
)

qa_template = """You are a highly strict and professional Malaysian Industrial Relations legal assistant. 

CONTEXT:
{context}

USER QUESTION:
{question}

CRITICAL RULES FOR ANSWERING:
1. STRICTLY GROUNDED: Base your answer EXCLUSIVELY on the provided CONTEXT. NEVER use outside knowledge.
2. CITATION MANDATORY: Cite the source at the end of the relevant sentence.
3. JURISDICTION FIRST (CRITICAL): If the user explicitely states they are a civil servant or government employee, you MUST evaluate if Act 177 applies to them based on the context (e.g., Section 52) BEFORE giving any other legal advice.
4. WORKMAN STATUS EXCLUSION (CRITICAL): If the user explicitly states they are a "freelancer" or "independent contractor", you MUST state that Act 177 Section 20 only protects a "workman" (an employee under a contract of service). DO NOT calculate backwages or provide Section 20 dispute procedures for independent contractors.
5. BE HIGHLY CONCISE: Output the direct legal answer using clear, professional paragraphs or bullet points.
6. GRACEFUL FALLBACK (CRITICAL): If the context does not completely answer the user's situation because the query was too short or vague, DO NOT flatly refuse. Instead, provide general Act 177 advice based on the closest context, and kindly ask the user to clarify their specific situation (e.g., "Based on the Act, an employee can claim unfair dismissal. Could you share more details about your termination?").
7. STRICT TIMELINE FOR DISMISSAL (MANDATORY): Under Act 177 Section 20, the strict legal time limit to file a representation for unfair dismissal is EXACTLY 60 DAYS. NEVER mention "30 days" or any other timeframe for filing a Section 20 representation.
8. SECTION ANCHORING: Whenever a user asks about being fired, dismissed, terminated, or sacked, you MUST explicitly name "Section 20 of the Industrial Relations Act 1967" as the specific statutory section giving them the right to claim.
9. STRICT FACT-MATCHING GUARDRAIL: If the retrieved cases do not contain the specific facts mentioned by the user (e.g., pregnancy, Covid), you can explain the general legal principles from the context, but DO NOT falsely claim that the retrieved cases involve those specific facts. State clearly that the context does not provide a direct case matching the exact scenario.
10. CONSTRUCTIVE DISMISSAL LOGIC: If the user describes being forced to quit due to the employer's unreasonable behavior or breach of contract, you MUST define this as "Constructive Dismissal" and state clearly that it IS considered a dismissal under Section 20 of Act 177.
FORMAT REQUIREMENT:
- Do NOT include ANY markdown headings (e.g., do NOT use "**Issue Identification**", "**Applicable Law**", etc.).
- Start your answer DIRECTLY with the core analysis and relevant legal rules.
- Maintain a clean and professional layout using direct sentences or standard bullet points.
- Do NOT include ANY markdown headings.
- NEVER cite or mention the internal prompt numbers (like "Rule 4" or "Rule 7") in your final output. ONLY cite the actual Act 177 sections or case names.
- STRICT NAMING CONVENTION (CRITICAL): Always refer to the Act EXACTLY as "Industrial Relations Act 1967 (Act 177)". NEVER write "Employment Act 177".

Ensure the name of the Act is always 'Industrial Relations Act 1967' (Act 177).”
Answer:"""
qa_chain = ChatPromptTemplate.from_template(qa_template) | llm | StrOutputParser()


# 4. LLM-as-a-Judge Evaluation (Backend Telemetry)
def evaluate_rag_turn(question: str, context: str, answer: str, latency: float):
    print("\n---Evaluation---")
    try:
        # --------------------------------------------------
        # A. Retrieval Evaluation
        # --------------------------------------------------
        # 1. Context Relevance (Proxy for Precision)
        rel_chain = ChatPromptTemplate.from_template(
            "Evaluate CONTEXT RELEVANCE. How relevant is the retrieved Context to the User Question? Does it contain mostly useful info without too much useless noise?\n"
            "Question: {question}\nContext: {context}\n\n"
            "CRITICAL INSTRUCTION: You MUST return ONLY a valid JSON object. Do not include markdown formatting.\n"
            "Format exactly like this:\n"
            "{{\"score\": 1.0, \"reason\": \"brief explanation\"}}"
        ) | judge_llm | JsonOutputParser()

        # 2. Context Sufficiency (Proxy for Recall)
        suf_chain = ChatPromptTemplate.from_template(
            "Evaluate CONTEXT SUFFICIENCY. Based ONLY on the retrieved Context, is there enough information to fully and accurately answer the User Question without external knowledge?\n"
            "Question: {question}\nContext: {context}\n\n"
            "CRITICAL INSTRUCTION: You MUST return ONLY a valid JSON object. Do not include markdown formatting.\n"
            "Format exactly like this:\n"
            "{{\"score\": 1.0, \"reason\": \"brief explanation\"}}"
        ) | judge_llm | JsonOutputParser()

        # --------------------------------------------------
        # B. Generation Evaluation
        # --------------------------------------------------
        # 3. Truthfulness (Faithfulness)
        f_chain = ChatPromptTemplate.from_template(
            "Evaluate faithfulness. Return a JSON object with 'score' (0 or 1) and 'reason'. "
            "EXCEPTION RULE: Do NOT penalize the answer (score it 1) for including general statutory knowledge like '1967' for Act 177, "
            "or the '60-day limit' and 'Section 20', as these are acceptable system-injected facts. "
            "Context: {context} Answer: {answer}"
        ) | judge_llm | JsonOutputParser()

        # 4. Completeness
        c_chain = ChatPromptTemplate.from_template(
            "Evaluate COMPLETENESS. Does the Answer address all parts of the Question?\n"
            "Question: {question}\nAnswer: {answer}\n\n"
            "CRITICAL INSTRUCTION: You MUST return ONLY a valid JSON object. Do not include markdown formatting.\n"
            "Format exactly like this:\n"
            "{{\"score\": 1.0, \"reason\": \"brief explanation\"}}"
        ) | judge_llm | JsonOutputParser()

        # 5. Conclusion Consistency
        con_chain = ChatPromptTemplate.from_template(
            "Evaluate CONCLUSION CONSISTENCY. Does the final legal advice logically align with the provided legal context?\n"
            "Context: {context}\nAnswer: {answer}\n\n"
            "CRITICAL INSTRUCTION: You MUST return ONLY a valid JSON object. Do not include markdown formatting.\n"
            "Format exactly like this:\n"
            "{{\"score\": 1.0, \"reason\": \"brief explanation\"}}"
        ) | judge_llm | JsonOutputParser()

        # Run all evaluations
        rel_result = rel_chain.invoke({"question": question, "context": context})
        suf_result = suf_chain.invoke({"question": question, "context": context})
        f_result = f_chain.invoke({"context": context, "answer": answer})
        c_result = c_chain.invoke({"question": question, "answer": answer})
        con_result = con_chain.invoke({"context": context, "answer": answer})

        rel_score = float(rel_result.get('score', 1.0))
        suf_score = float(suf_result.get('score', 1.0))
        f_score = float(f_result.get('score', 1.0))
        c_score = float(c_result.get('score', 1.0))
        con_score = float(con_result.get('score', 1.0))

        error_category = "None"
        error_reason = ""

        if f_score < 1 or c_score < 0.8 or con_score < 1 or rel_score < 0.5 or suf_score < 0.5:
            err_chain = ChatPromptTemplate.from_template(
                "Perform Error Taxonomy Analysis. Categorize the failure into EXACTLY ONE of these categories:\n"
                "1. 'Hallucination' (Answer contains facts not in Context)\n"
                "2. 'Incomplete Retrieval' (Key legal facts needed to answer are missing from Context)\n"
                "3. 'Wrong Interpretation' (Law was misapplied or conclusion is illogical)\n"
                "4. 'Irrelevant Context' (Context is entirely unrelated to the Question)\n"
                "5. 'None' (False alarm, the answer is actually correct)\n"
                "Context: {context}\nQuestion: {question}\nAnswer: {answer}\n\n"
                "CRITICAL INSTRUCTION: You MUST return ONLY a valid JSON object. Do not include markdown formatting.\n"
                "Format exactly like this:\n"
                "{{\"category\": \"None\", \"reason\": \"brief explanation\"}}"
            ) | judge_llm | JsonOutputParser()
            
            err_result = err_chain.invoke({"context": context, "question": question, "answer": answer})
            error_category = err_result.get('category', 'Unknown')
            error_reason = err_result.get('reason', 'N/A')

        # Print log (terminal display)
        print(f"    Latency: {latency:.2f}s")
        print(f"    Retrieval > Relevance (Precision): {rel_score} | Sufficiency (Recall): {suf_score}")
        print(f"    Generation > Truthfulness: {f_score} | Completeness: {c_score} | Consistency: {con_score}")
        
        if error_category != "None":
            print(f"⚠️  Error Detected: {error_category} - {error_reason}")
        print("-" * 55)

        # Save to Metrics Database
        conn = sqlite3.connect("metrics.db")
        c = conn.cursor()
        c.execute('''
            INSERT INTO rag_evaluations 
            (user_query, latency, truthfulness_score, completeness_score, consistency_score, context_relevance_score, context_sufficiency_score, error_category, error_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (question, latency, f_score, c_score, con_score, rel_score, suf_score, error_category, error_reason))
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")


# 5. Defining API endpoints
class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
async def ask_legal_assistant(request: AskRequest, background_tasks: BackgroundTasks):
    start_time = time.time()

    user_query = request.question
    clean_query = user_query.strip().lower().rstrip(".?/")

    # 1. Refine Query
    refined_keywords = refine_chain.invoke({"original_query": clean_query})
    print(f"\n[API Request] Query: {clean_query} -> Keywords: {refined_keywords}")

    # If determined to be small talk or a vague request for help, proceed directly to the fallback.
    if "TRIGGER_FALLBACK" in refined_keywords:
        fallback_template = """You are an empathetic and professional Malaysian legal assistant. 
        The user typed: '{question}'. 
        Respond warmly in 2-3 sentences. Introduce yourself as an AI legal assistant for Act 177, express readiness to help, and gently prompt them to share specific details about their workplace issue (e.g., dismissal, salary, resignation). DO NOT mention any internal rules."""
        
        fallback_chain = ChatPromptTemplate.from_template(fallback_template) | llm | StrOutputParser()
        response = fallback_chain.invoke({"question": clean_query})
        
        return {
            "status": "success",
            "answer": response.replace("\n", "<br>"),
            "citations": [],
            "confidence": 0
        }
    
    # 2. Retrieve & Rerank
    docs_act = retriever_act.invoke(refined_keywords)
    docs_cases = retriever_cases.invoke(refined_keywords)

    scored_act = sorted(
        zip(
            rerank_model.predict(
                [[refined_keywords, d.page_content] for d in docs_act]
            ),
            docs_act,
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    final_act = [doc for score, doc in scored_act if score > -0.5][:3]

    print("\n[Reranking Scores](Act 177 Sections)")
    for score, doc in scored_act:
        source = doc.metadata.get("source", "Unknown Section")
        print(f">Score: {score:>7.4f} | {source}")

    scored_cases = sorted(
        zip(
            rerank_model.predict(
                [[refined_keywords, d.page_content] for d in docs_cases]
            ),
            docs_cases,
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    print("\n[Reranking Scores](Cases)")
    for score, doc in scored_cases:
        source = doc.metadata.get("source", "Unknown Case")
        # If a case name is too long, truncate it to the first 60 characters in the terminal to keep the layout neat
        display_source = source[:60] + "..." if len(source) > 60 else source
        print(f">Score: {score:>7.4f} | {display_source}")
    print("-" * 55)

    final_cases = [doc for score, doc in scored_cases if score > -1.0][:2]

    # 3. Build Citations format for frontend
    citations_for_ui = []
    cit_num = 1

    # 4. Generate Response
    if not final_act:
        # Provide gentle guidance rather than displaying an error message straight away
        fallback_template = """You are a polite receptionist for a Malaysian Industrial Relations (Act 177) legal assistant. 
        The user typed: '{question}'. 
        
        CRITICAL RULES FOR RECEPTIONIST:
        1. NO LEGAL ADVICE (ABSOLUTE RULE): You do not have access to the law database right now. You MUST NOT give any legal advice, MUST NOT mention any specific laws, MUST NOT judge if an action is "just cause", and MUST NOT promise to help with specific claims like overtime or defamation.
        2. IF USER ASKS A COMPLEX LEGAL QUESTION: Acknowledge their situation politely, but tell them their query is too complex for your search engine. Ask them to summarize their main legal issue in a few short keywords (e.g., "dismissed due to pregnancy" or "resigned because salary cut").
        3. IF GREETING/CHAT: Introduce yourself as an Act 177 AI assistant and ask for keywords related to their workplace dispute.
        
        Respond warmly in 2-3 sentences."""
        
        fallback_chain = ChatPromptTemplate.from_template(fallback_template) | llm | StrOutputParser()
        response = fallback_chain.invoke({"question": clean_query})
        avg_confidence = 0
        citations_for_ui = []

        context_text = "No legal context retrieved (Fallback triggered)."
    else:
        context_text = "=== STATUTORY LAW (ACT 177) ===\n"
        for doc in final_act:
            source = doc.metadata.get("source", "Unknown Section")
            context_text += f"--- [Source: {source}] ---\n{doc.page_content}\n\n"
            citations_for_ui.append(
                {
                    "num": cit_num,
                    "doc": source,
                    "excerpt": doc.page_content[:120] + "...",
                }
            )
            cit_num += 1

        if final_cases:
            context_text += "=== CASE LAW (PAST CASES) ===\n"
            for doc in final_cases:
                source = doc.metadata.get("source", "Unknown Case")
                context_text += f"--- [Source: {source}] ---\n{doc.page_content}\n\n"
                citations_for_ui.append(
                    {
                        "num": cit_num,
                        "doc": source,
                        "excerpt": doc.page_content[:120] + "...",
                    }
                )
                cit_num += 1 

        response = qa_chain.invoke({"context": context_text, "question": clean_query})

        #  Calculate a rough confidence score to display on the front end (based on the mapping of the highest Rerank score)
        highest_score = scored_act[0][0] if scored_act else 0
        avg_confidence = min(99, max(60, int(highest_score * 10) + 75))

    # Calculate the latency 
    end_time = time.time()
    latency = end_time - start_time 

    #  Run the assessment as a background task, so that the front end can display the answers without having to wait for the assessment to finish
    background_tasks.add_task(
        evaluate_rag_turn, clean_query, context_text, response, latency
    )

    html_response = response.replace("\n", "<br>")
    html_response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_response)
    # 5. Return to HTML
    return {
        "status": "success",
        "answer": html_response,
        "citations": citations_for_ui,
        "confidence": avg_confidence,
        "latency_sec": round(latency, 2)
    }

# 6. Analytics Dashboard API (FYP Evaluation Display)
@app.get("/api/metrics")
async def get_system_metrics():
    try:
        conn = sqlite3.connect("metrics.db")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Retrieve overall aggregated data
        c.execute('''
            SELECT 
                COUNT(*) as total_queries,
                AVG(latency) as avg_latency,
                AVG(truthfulness_score) as avg_truthfulness,
                AVG(completeness_score) as avg_completeness,
                AVG(consistency_score) as avg_consistency,
                AVG(context_relevance_score) as avg_relevance,
                AVG(context_sufficiency_score) as avg_sufficiency
            FROM rag_evaluations
        ''')
        overall = dict(c.fetchone())
        
        # Retrieve the Error Taxonomy distribution
        c.execute('''
            SELECT error_category, COUNT(*) as count 
            FROM rag_evaluations 
            WHERE error_category != 'None' 
            GROUP BY error_category
        ''')
        errors = [dict(row) for row in c.fetchall()]
        
        conn.close()
        return {
            "status": "success",
            "overall_metrics": overall,
            "error_taxonomy": errors
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}