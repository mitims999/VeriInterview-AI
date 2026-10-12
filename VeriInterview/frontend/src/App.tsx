import { useState } from "react";
import type { FullReport } from "./api";
import type { SmartReport } from "./api";
import UploadForm from "./components/UploadForm";
import SmartUploadForm from "./components/SmartUploadForm";
import ReportView from "./components/ReportView";
import SmartReportView from "./components/SmartReportView";
import SavedInterviews from "./components/SavedInterviews";
import type { SavedInterview } from "./components/SavedInterviews";
import savedData from "./data/completed-interviews.json";
import Icon from "./components/Icon";

const savedInterviews = savedData as SavedInterview[];
type Mode = "upload" | "smart" | "completed";

export default function App() {
  const [mode, setMode] = useState<Mode>("smart");
  const [selected, setSelected] = useState("");
  const [uploadReport, setUploadReport] = useState<FullReport | null>(null);
  const [smartReport, setSmartReport] = useState<SmartReport | null>(null);
  const [guide, setGuide] = useState(false);
  const [formKey, setFormKey] = useState(0);
  const savedReport = savedInterviews.find(interview => interview.id === selected)?.report;
  const report = mode === "completed" ? savedReport : mode === "smart" ? smartReport : uploadReport;

  function showCompleted() { setMode("completed"); setSelected(""); window.scrollTo({ top: 0, behavior: "smooth" }); }
  function newReview() { setMode("upload"); setUploadReport(null); setFormKey(k => k + 1); window.scrollTo({ top: 0, behavior: "smooth" }); }
  function newSmartReview() { setMode("smart"); setSmartReport(null); setFormKey(k => k + 1); window.scrollTo({ top: 0, behavior: "smooth" }); }
  function openReport(id: string) { setSelected(id); window.scrollTo({ top: 0, behavior: "smooth" }); }

  const headingText = (() => {
    if (report) return report.assessment.candidate_name;
    if (mode === "smart") return "Resume vs. transcript intelligence.";
    if (mode === "upload") return "Good interviews start with clarity.";
    return "A few conversations to explore.";
  })();

  const subText = (() => {
    if (report) return "Review the evidence, understand the context, and decide what to ask next.";
    if (mode === "smart") return "Upload a resume and an interview transcript — we'll check if the candidate actually knows what they claim.";
    if (mode === "upload") return "Bring the resume and conversation together for a thoughtful, evidence-led review.";
    return "Start with three completed sample interviews and see what a full review looks like.";
  })();

  const eyebrow = (() => {
    if (report && mode === "smart") return "SMART ANALYSIS";
    if (report) return "INTERVIEW REVIEW";
    if (mode === "smart") return "RESUME INTELLIGENCE";
    if (mode === "upload") return "A MORE INFORMED CONVERSATION";
    return "COMPLETED INTERVIEWS";
  })();

  const breadcrumb = (() => {
    if (report) return "Interview report";
    if (mode === "smart") return "Smart analysis";
    if (mode === "upload") return "New review";
    return "Completed interviews";
  })();

  return <div className="workspace">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); newSmartReview(); }} aria-label="VeriInterview home"><span className="brand-mark"><Icon name="shield" size={24}/></span><span>veri<span className="brand-light">interview</span><small>INTERVIEWER WORKSPACE</small></span></a>
      <div className="nav-label">WORKSPACE</div>
      <nav aria-label="Main navigation">
        <button className={`nav-item ${mode === "smart" ? "active" : ""}`} onClick={newSmartReview}><Icon name="shield"/>Smart analysis<span className="nav-badge">NEW</span></button>
        <button className={`nav-item ${mode === "upload" ? "active" : ""}`} onClick={() => setMode("upload")}><Icon name="plus"/>New review<span className="nav-dot"/></button>
        <button className={`nav-item ${mode === "completed" ? "active" : ""}`} onClick={showCompleted}><Icon name="people"/>Completed interviews<span className="nav-count">{savedInterviews.length}</span></button>
      </nav>
      <div className="sidebar-note"><span className="small-icon"><Icon name="shield" size={18}/></span><h3>Evidence. Then judgment.</h3><p>A clearer view of interview integrity. The final decision is always yours.</p></div>
      <div className="sidebar-bottom"><button className="nav-item" onClick={() => setGuide(v => !v)} aria-expanded={guide}><Icon name="help"/>How it works</button><div className="workspace-profile"><span className="avatar">IW</span><span>Interviewer<small>Local workspace</small></span><span className="status-dot" title="Local workspace"/></div></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb">Workspace<span>/</span><strong>{breadcrumb}</strong></div></header>
      <main className="page">
        {mode === "completed" && report && <button className="text-button back-to-interviews" onClick={showCompleted}><Icon name="arrow" size={15} style={{ transform: "rotate(180deg)" }}/>All completed interviews</button>}
        <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{headingText}</h1><p>{subText}</p></div>{(report || mode === "completed") && <button className="button secondary" onClick={newSmartReview}><Icon name="shield" size={17}/>Smart analysis</button>}</div>
        {guide && <section className="guide-panel" aria-label="How it works"><div className="section-heading"><h2>From conversation to context</h2><button className="icon-button" aria-label="Close guide" onClick={() => setGuide(false)}><Icon name="close"/></button></div><div className="guide-grid"><div><b>01 / Prepare</b><p>Add a resume and transcript. For smart analysis, just two files are needed — no formatting required.</p></div><div><b>02 / Review</b><p>Compare demonstrated skills with resume claims. See keyword intelligence and cross-reference results.</p></div><div><b>03 / Follow up</b><p>Use flagged evidence to guide a verification conversation. Scores support your judgment.</p></div></div></section>}
        {mode === "smart" && !smartReport && <SmartUploadForm key={formKey} onResult={r => { setSmartReport(r); window.scrollTo({ top: 0, behavior: "smooth" }); }}/>}
        {mode === "upload" && !uploadReport && <UploadForm key={formKey} onResult={r => { setUploadReport(r); window.scrollTo({ top: 0, behavior: "smooth" }); }} onDemo={showCompleted}/>}
        {mode === "completed" && !report && <SavedInterviews interviews={savedInterviews} onOpen={openReport}/>}
        {mode === "smart" && smartReport && <SmartReportView key="smart" report={smartReport}/>}
        {mode === "upload" && uploadReport && <ReportView key="upload" report={uploadReport}/>}
        {mode === "completed" && report && <ReportView key={selected} report={report}/>}
        <footer className="page-footer"><span><Icon name="shield" size={14}/> Built for human review</span><span>VeriInterview · TCS Tech Day</span></footer>
      </main>
    </div>
  </div>;
}
