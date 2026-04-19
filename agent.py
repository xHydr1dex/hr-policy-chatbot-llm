# agent.py - NovaTech Solutions HR Policy Bot
# Standalone module. Import build_agent() in capstone_streamlit.py

import os, datetime
from dotenv import load_dotenv
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

DOCUMENTS = [
    {"id": "doc_001", "topic": "Annual Leave Policy", "text": 'NovaTech Solutions grants every confirmed full-time employee 20 days of paid annual leave per calendar year. Leave accrues at 1.67 days per month from the date of joining. Applications must be submitted via the HR portal at least 5 working days in advance. Up to 10 unused annual leave days may be carried forward; the remainder lapses on December 31st. Employees need 6 months of continuous service before taking annual leave. Leave during probation is treated as Loss of Pay. Annual leave cannot be encashed during employment — only at full and final settlement. Public holidays falling within a leave period are not counted as leave days. Leaves exceeding 10 consecutive working days require written approval from the department head and HR.'},
    {"id": "doc_002", "topic": "Casual Leave and Sick Leave", "text": 'NovaTech Solutions provides 8 days of casual leave (CL) and 8 days of sick leave (SL) per calendar year to all full-time employees. Casual leave is for short unplanned personal matters; apply on the day of absence or the previous evening via the HR portal. Sick leave requires a medical certificate for absences exceeding 2 consecutive days. CL and SL cannot be carried forward, clubbed with annual leave, or encashed. Maximum 3 consecutive casual leave days allowed at once; beyond this, apply for annual leave. Sick leave may be taken in half-day units with manager approval. Both CL and SL reset on January 1st each year. Mid-year joiners get a pro-rated allocation. Unused CL and SL are forfeited at year-end with no encashment.'},
    {"id": "doc_003", "topic": "Remote Work and Flexible Hours", "text": 'NovaTech Solutions is a remote-first company. All full-time employees may work from home by default with no monthly WFH cap. Core hours are 10 AM to 4 PM IST, Monday to Friday. NovaTech provides a one-time home office setup allowance of Rs 15000 after 3 months of service. Co-working space subscriptions up to Rs 5000 per month are reimbursable with invoices. The Innovation Time policy entitles each employee to 10 percent of weekly hours (about 4 hours) for internal projects or skill development, with manager awareness. The Deep Work Block from 9 AM to 12 PM is a no-internal-meeting window; all internal meetings must be scheduled between 12 PM and 5 PM. Employees may work from an international travel location for up to 4 weeks per year without affecting their leave balance, subject to manager notification.'},
    {"id": "doc_004", "topic": "Payroll and Compensation", "text": 'Salaries at NovaTech are credited on the last working day of every month via direct bank transfer. CTC breakdown: Basic Salary 40 percent, HRA 20 percent, Special Allowance variable, PF employer contribution 12 percent of Basic, Performance Bonus up to 15 percent of annual CTC. TDS is deducted monthly; employees must submit investment proofs (Form 12BB) by January 31st to avoid excess deduction in Q4. Salary slips are available in the HR portal by the 2nd of each following month. Variable Pay is disbursed quarterly in April, July, October, and January, subject to achieving at least 80 percent of quarterly OKRs. NovaTech issues Form 16 to all employees by May 31st every year for income tax filing. A joining bonus, if applicable, carries a 12-month clawback clause.'},
    {"id": "doc_005", "topic": "Performance Review and Appraisal", "text": 'NovaTech follows a bi-annual review cycle: Mid-Year Review in July and Annual Appraisal in January. Performance is OKR-based (Objectives and Key Results). Employees set OKRs each half-year with their manager in the first week of the cycle. Rating scale: Exceptional (5), Exceeds Expectations (4), Meets Expectations (3), Needs Improvement (2), Unsatisfactory (1). Annual increments: Exceptional 20-25 percent, Exceeds Expectations 12-18 percent, Meets Expectations 8-10 percent, Needs Improvement 0-4 percent, Unsatisfactory: no increment + PIP. PIP period is 60 days with bi-weekly check-ins. Promotions require minimum 12 months in current role. Final rating is 60 percent manager evaluation and 40 percent 360-degree feedback.'},
    {"id": "doc_006", "topic": "Learning and Development Budget", "text": 'Every full-time employee receives an annual L&D budget of Rs 50000 per financial year (April 1 to March 31). Eligible uses: online courses (Udemy, Coursera, LinkedIn Learning), certifications (AWS, GCP, Azure, PMP), conferences, technical books, and bootcamps. Employees need 6 months of service before accessing the L&D budget. Claims must be submitted within 30 days of course completion with invoice and certificate. Unused amounts lapse on March 31st and do not carry forward. Prior approval from HR and department head is required for expenses above Rs 25000. A proportional clawback applies if the employee resigns within 6 months of claiming above Rs 10000. NovaTech also organizes monthly NovaTalks: 45-minute employee-led knowledge-sharing sessions.'},
    {"id": "doc_007", "topic": "Health Insurance and Benefits", "text": 'All full-time employees are enrolled in group health insurance from Day 1. Coverage: Employee and spouse Rs 500000 per person per year; two dependent children Rs 250000 per child; parents or parents-in-law (optional) Rs 300000 at a subsidized premium of Rs 6000 per year. Policy covers hospitalization, day-care, pre-hospitalization 60 days, post-hospitalization 90 days, and maternity up to Rs 75000. Pre-existing diseases are covered after a 2-year waiting period. Cashless treatment at over 8000 network hospitals across India. Up to 6 mental health counseling sessions per year with empaneled psychologists are covered. Dental and vision are not in the standard policy but available as an add-on at Rs 3000 per year. The insurance year runs April 1 to March 31.'},
    {"id": "doc_008", "topic": "Probation and Confirmation", "text": 'All new employees serve a mandatory 3-month probation period from their date of joining. During probation, employees receive full salary and health insurance coverage. Leaves taken during probation are treated as Loss of Pay since accrual does not apply retroactively. The L&D budget is not accessible during the first 6 months of employment. A performance review is conducted at the end of months 2 and 3. A confirmation letter is issued within 2 weeks of successful completion. Probation may be extended by up to 3 additional months if performance is unsatisfactory; failure during extension results in termination with 30 days notice. During probation the resignation notice period is 2 weeks instead of the standard 60 days. Probation may be waived for lateral hires with 5 or more years of directly relevant experience.'},
    {"id": "doc_009", "topic": "Resignation and Exit Process", "text": 'Employees wishing to resign must submit a formal resignation via the HR portal or email to their manager and HR. The notice period for all confirmed employees is 60 days (2 months). During notice, the employee must complete deliverables, produce knowledge transfer documentation, and hand over responsibilities. Notice buyout requires approval from the department head and HR Director and is not available for employees in critical project deliveries within 30 days of a major release. Full and Final (F&F) settlement is processed within 45 working days of the last working day, including remaining salary, earned leave encashment, pro-rata variable pay, and reimbursements. Relieving letter and experience letter are issued within 5 working days of F&F settlement. All company assets (laptop, access cards) must be returned on the last working day.'},
    {"id": "doc_010", "topic": "Employee Referral Bonus", "text": 'Referral bonuses are paid in two tranches: 50 percent on the referred candidate joining and 50 percent after the candidate completes 3 months of employment. Bonus amounts by level: Intern or Fresher Rs 5000; Junior Engineer L1-L2 Rs 15000; Senior Engineer L3-L4 Rs 30000; Lead or Manager L5-L6 Rs 50000; Director and above Rs 75000. A Super Referral bonus of up to Rs 100000 may be approved by the CEO for exceptional hires. Referrals must be submitted via the HR portal before the candidate applies independently. The referring employee must be active (not serving notice) at the time of both tranche payments. Candidates who worked at NovaTech within the last 2 years are not eligible. There is no cap on the number of referrals per employee. Open roles are posted every Monday on #talent-hunt.'},
    {"id": "doc_011", "topic": "Mental Wellness and Mental Health Days", "text": 'Every employee is entitled to 4 dedicated Mental Health Days per calendar year, completely separate from sick leave, casual leave, and annual leave. Mental health days require no documentation, no medical certificate, and no advance notice; a same-day message to the reporting manager is sufficient. NovaTech partners with Wysa (AI wellness app) and Vandrevala Foundation (human counseling) to provide free, confidential mental health support for all employees and their immediate families. The EAP helpline is 1860-2662-345, confidential and not shared with the employer. Up to 6 licensed therapist sessions per year are covered under the group insurance. Employees returning from mental health leave longer than 5 consecutive days receive a phased return-to-work plan with reduced workload for up to 2 weeks. Discussing mental health with HR or a manager is protected and cannot be grounds for adverse decisions.'},
    {"id": "doc_012", "topic": "Code of Conduct and Anti-Harassment", "text": 'NovaTech maintains zero tolerance against harassment, discrimination, and misconduct. The policy applies to all employees, contractors, interns, and vendors during work hours, at company events, and on digital platforms including Slack, email, and video calls. Prohibited conduct includes verbal abuse, sexual harassment, bullying, and discrimination based on gender, religion, caste, nationality, disability, sexual orientation, or age. Report to the Internal Complaints Committee (ICC) at icc@novatech.in or via the anonymous form on the HR portal. The ICC is constituted under the POSH Act 2013 and must complete inquiries within 90 days. Retaliation against a complainant is itself a disciplinary offense. All employees must complete the mandatory POSH e-module within their first 2 weeks of joining. Annual POSH refresher training is conducted every December for the entire organization.'},
]

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    employee_id: str
    department: str


def build_agent():
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client   = chromadb.Client()
    try: client.delete_collection("novatech_hr_kb")
    except Exception: pass
    collection = client.create_collection("novatech_hr_kb")
    texts = [d["text"] for d in DOCUMENTS]
    collection.add(documents=texts, embeddings=embedder.encode(texts).tolist(),
                   ids=[d["id"] for d in DOCUMENTS],
                   metadatas=[{"topic": d["topic"]} for d in DOCUMENTS])

    def memory_node(state):
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        return {"messages": msgs[-6:]}

    def router_node(state):
        q = state["question"]; msgs = state.get("messages", [])
        recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in msgs[-3:-1]) or "none"
        prompt = (
            "You are a router for Aria, NovaTech HR Assistant.\n"
            "Options:\n"
            "- retrieve: HR policy questions (leave, payroll, benefits, appraisal, probation, exit, referral, wellness, conduct)\n"
            "- memory_only: follow-up about what was just said\n"
            "- tool: date-based calculations (leave balance, notice period end, upcoming deadlines)\n"
            f"Conversation: {recent}\nQuestion: {q}\nReply ONE word: retrieve / memory_only / tool"
        )
        d = llm.invoke(prompt).content.strip().lower()
        if "memory" in d: d = "memory_only"
        elif "tool" in d: d = "tool"
        else:             d = "retrieve"
        return {"route": d}

    def retrieval_node(state):
        res    = collection.query(query_embeddings=embedder.encode([state["question"]]).tolist(), n_results=3)
        topics = [m["topic"] for m in res["metadatas"][0]]
        ctx    = "\n\n---\n\n".join(f"[{topics[i]}]\n{res['documents'][0][i]}" for i in range(len(topics)))
        return {"retrieved": ctx, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        q = state["question"].lower(); today = datetime.date.today()
        try:
            if any(w in q for w in ["notice period", "last working day", "resign"]):
                end = today + datetime.timedelta(days=60)
                r = f"Resignation today ({today}) -> notice ends {end.strftime('%A, %B %d, %Y')} (60 days)."
            elif any(w in q for w in ["accrued", "leave balance", "leave days", "how many leave", "how much leave"]):
                acc = round(min(20.0, ((today.month-1) + today.day/30.0) * (20.0/12.0)), 1)
                r   = (f"As of {today}: Annual Leave accrued = {acc}/20 days. "
                       f"CL = 8/yr. SL = 8/yr. Mental Health Days = 4/yr. "
                       f"Check HR portal for personal balance after leaves taken.")
            elif any(w in q for w in ["deadline", "upcoming", "next hr"]):
                yr = today.year
                ms = [(datetime.date(yr,1,31),"Form 12BB deadline"),
                      (datetime.date(yr,3,31),"L&D budget lapses"),
                      (datetime.date(yr,4,1), "Insurance year renews"),
                      (datetime.date(yr,7,1), "Mid-year review opens"),
                      (datetime.date(yr,12,31),"CL/SL balances lapse")]
                coming = sorted([(d,l) for d,l in ms if d>=today])[:4]
                r = "Upcoming HR deadlines:\n" + "\n".join(f"  {d} ({(d-today).days}d) — {l}" for d,l in coming)
            else:
                r = f"Today is {today.strftime('%A, %B %d, %Y')} (Q{(today.month-1)//3+1} {today.year})."
        except Exception as e:
            r = f"Calculator error: {e}. Contact hr@novatech.in."
        return {"tool_result": r}

    def answer_node(state):
        q=state["question"]; ret=state.get("retrieved",""); tool=state.get("tool_result","")
        msgs=state.get("messages",[]); retries=state.get("eval_retries",0)
        parts = []
        if ret:  parts.append(f"POLICY KB:\n{ret}")
        if tool: parts.append(f"CALCULATOR:\n{tool}")
        ctx = "\n\n".join(parts)
        if ctx:
            sys_msg = (f"You are Aria, NovaTech HR Assistant. Answer using ONLY the context below. "
                       f"If not in context say: I do not have that information. Contact hr@novatech.in.\n\n{ctx}")
        else:
            sys_msg = "You are Aria, NovaTech HR Assistant. Answer from conversation history or ask to rephrase."
        if retries > 0:
            sys_msg += "\n\nPrevious answer failed quality check. Use ONLY context above."
        lc = [SystemMessage(content=sys_msg)]
        for m in msgs[:-1]:
            lc.append(HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]))
        lc.append(HumanMessage(content=q))
        return {"answer": llm.invoke(lc).content}

    def eval_node(state):
        ans=state.get("answer",""); ctx=state.get("retrieved","")[:500]; r=state.get("eval_retries",0)
        if not ctx: return {"faithfulness": 1.0, "eval_retries": r+1}
        try:
            score = float(llm.invoke(
                f"Rate faithfulness 0.0-1.0 (number only).\nContext:{ctx}\nAnswer:{ans[:300]}"
            ).content.strip().split()[0])
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5
        return {"faithfulness": score, "eval_retries": r+1}

    def save_node(state):
        return {"messages": state.get("messages",[]) + [{"role":"assistant","content":state["answer"]}]}

    def route_decision(state):
        r = state.get("route", "retrieve")
        return "tool" if r=="tool" else "skip" if r=="memory_only" else "retrieve"

    def eval_decision(state):
        return "save" if (state.get("faithfulness",1.0)>=FAITHFULNESS_THRESHOLD or
                          state.get("eval_retries",0)>=MAX_EVAL_RETRIES) else "answer"

    g = StateGraph(CapstoneState)
    for name, fn in [("memory",memory_node),("router",router_node),("retrieve",retrieval_node),
                     ("skip",skip_retrieval_node),("tool",tool_node),("answer",answer_node),
                     ("eval",eval_node),("save",save_node)]:
        g.add_node(name, fn)
    g.set_entry_point("memory")
    g.add_edge("memory","router")
    g.add_conditional_edges("router", route_decision, {"retrieve":"retrieve","skip":"skip","tool":"tool"})
    g.add_edge("retrieve","answer"); g.add_edge("skip","answer"); g.add_edge("tool","answer")
    g.add_edge("answer","eval")
    g.add_conditional_edges("eval", eval_decision, {"answer":"answer","save":"save"})
    g.add_edge("save", END)
    app = g.compile(checkpointer=MemorySaver())
    return app, embedder, collection
