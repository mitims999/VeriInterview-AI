import { useState } from "react";
import type { FullReport } from "./api";
import UploadForm from "./components/UploadForm";
import ReportView from "./components/ReportView";
import SavedInterviews from "./components/SavedInterviews";
import type { SavedInterview } from "./components/SavedInterviews";
import savedData from "./data/completed-interviews.json";
import Icon from "./components/Icon";

const savedInterviews = savedData as SavedInterview[];
type Mode = "upload" | "completed";

export default function App() {
  const [mode, setMode] = useState<Mode>("upload");
  const [selected, setSelected] = useState("");
  const [uploadReport, setUploadReport] = useState<FullReport | null>(null);
  const [guide, setGuide] = useState(false);
  const [formKey, setFormKey] = useState(0);
  const savedReport = savedInterviews.find(interview => interview.id === selected)?.report;
  const report = mode === "completed" ? savedReport : uploadReport;

  function showCompleted() { setMode("completed"); setSelected(""); window.scrollTo({ top: 0, behavior: "smooth" }); }
  function newReview() { setMode("upload"); setUploadReport(null); setFormKey(k => k + 1); window.scrollTo({ top: 0, behavior: "smooth" }); }
  function openReport(id: string) { setSelected(id); window.scrollTo({ top: 0, behavior: "smooth" }); }

  return <div className="workspace">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); newReview(); }} aria-label="VeriInterview home"><span className="brand-mark"><Icon name="shield" size={24}/></span><span>veri<span className="brand-light">interview</span><small>INTERVIEWER WORKSPACE</small></span></a>
      <div className="nav-label">WORKSPACE</div>
      <nav aria-label="Main navigation">
        <button className={`nav-item ${mode === "upload" ? "active" : ""}`} onClick={() => setMode("upload")}><Icon name="plus"/>New review<span className="nav-dot"/></button>
        <button className={`nav-item ${mode === "completed" ? "active" : ""}`} onClick={showCompleted}><Icon name="people"/>Completed interviews<span className="nav-count">{savedInterviews.length}</span></button>
      </nav>
      <div className="sidebar-note"><span className="small-icon"><Icon name="shield" size={18}/></span><h3>Evidence. Then judgment.</h3><p>A clearer view of interview integrity. The final decision is always yours.</p></div>
      <div className="sidebar-bottom"><button className="nav-item" onClick={() => setGuide(v => !v)} aria-expanded={guide}><Icon name="help"/>How it works</button><div className="workspace-profile"><span className="avatar">IW</span><span>Interviewer<small>Local workspace</small></span><span className="status-dot" title="Local workspace"/></div></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb">Workspace<span>/</span><strong>{report ? "Interview report" : mode === "upload" ? "New review" : "Completed interviews"}</strong></div></header>
      <main className="page">
        {mode === "completed" && report && <button className="text-button back-to-interviews" onClick={showCompleted}><Icon name="arrow" size={15} style={{ transform: "rotate(180deg)" }}/>All completed interviews</button>}
        <div className="page-heading"><div><div className="eyebrow">{report ? "INTERVIEW REVIEW" : mode === "upload" ? "A MORE INFORMED CONVERSATION" : "COMPLETED INTERVIEWS"}</div><h1>{report ? report.assessment.candidate_name : mode === "upload" ? "Good interviews start with clarity." : "A few conversations to explore."}</h1><p>{report ? "Review the evidence, understand the context, and decide what to ask next." : mode === "upload" ? "Bring the resume and conversation together for a thoughtful, evidence-led review." : "Start with three completed sample interviews and see what a full review looks like."}</p></div>{(report || mode === "completed") && <button className="button secondary" onClick={newReview}><Icon name="plus" size={17}/>New review</button>}</div>
        {guide && <section className="guide-panel" aria-label="How it works"><div className="section-heading"><h2>From conversation to context</h2><button className="icon-button" aria-label="Close guide" onClick={() => setGuide(false)}><Icon name="close"/></button></div><div className="guide-grid"><div><b>01 / Prepare</b><p>Add a resume, reference answers, and a transcript. Sample documents are available in the form.</p></div><div><b>02 / Review</b><p>Compare demonstrated skills with resume claims, then read the evidence behind each signal.</p></div><div><b>03 / Follow up</b><p>Use flagged evidence to guide a verification conversation. Scores support your judgment.</p></div></div></section>}
        {mode === "upload" && !uploadReport && <UploadForm key={formKey} onResult={r => { setUploadReport(r); window.scrollTo({ top: 0, behavior: "smooth" }); }} onDemo={showCompleted}/>}
        {mode === "completed" && !report && <SavedInterviews interviews={savedInterviews} onOpen={openReport}/>}
        {report && <ReportView key={mode === "completed" ? selected : "upload"} report={report}/>}
        <footer className="page-footer"><span><Icon name="shield" size={14}/> Built for human review</span><span>VeriInterview · TCS Tech Day</span></footer>
      </main>
    </div>
  </div>;
}
