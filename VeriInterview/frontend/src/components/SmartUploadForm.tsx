import { useRef, useState } from "react";
import { smartAnalyze } from "../api";
import type { SmartReport } from "../api";
import { getSampleFiles } from "../sampleData";
import Icon from "./Icon";

function FileField({ label, hint, file, onChange, accept }: { label: string; hint: string; file: File | null; onChange: (file: File | null) => void; accept: string }) {
  const ref = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  function select(file: File | undefined) {
    if (!file) return;
    const allowed = accept.split(",").some(ext => file.name.toLowerCase().endsWith(ext));
    if (!allowed) { setError(`Please choose ${accept.replaceAll(",", ", ")} for this field.`); return; }
    setError(""); onChange(file);
  }
  return <div className="file-field"><div className="field-title">{label}<span>Required</span></div><p className="field-hint">{hint}</p><input ref={ref} className="sr-only" type="file" accept={accept} aria-label={label} onChange={e => select(e.target.files?.[0])} tabIndex={-1}/><div className={`dropzone ${file ? "has-file" : ""} ${dragging ? "dragging" : ""}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); select(e.dataTransfer.files[0]); }}>
    <button type="button" className="file-select" onClick={() => ref.current?.click()}><span className="file-icon"><Icon name={file ? "check" : "upload"} size={20}/></span><span className="file-description"><strong>{file ? file.name : "Click to upload or drop a file"}</strong><small>{file ? `${(file.size / 1024).toFixed(0)} KB · Click to replace` : accept.replaceAll(".", "").replaceAll(",", " / ").toUpperCase()}</small></span></button>{file && <button type="button" className="icon-button remove-file" aria-label={`Remove ${label}`} onClick={() => { onChange(null); if (ref.current) ref.current.value = ""; }}><Icon name="close" size={16}/></button>}
  </div>{error && <p className="field-error" role="alert">{error}</p>}</div>;
}

export default function SmartUploadForm({ onResult }: { onResult: (report: SmartReport) => void }) {
  const [candidateName, setCandidateName] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [resume, setResume] = useState<File | null>(null);
  const [transcript, setTranscript] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [formatHelp, setFormatHelp] = useState(false);
  const canSubmit = candidateName.trim() && resume && transcript && !loading;

  async function submit(e: React.FormEvent) {
    e.preventDefault(); if (!canSubmit || !resume || !transcript) return;
    setLoading(true); setError("");
    try { onResult(await smartAnalyze({ candidateName: candidateName.trim(), jobDescription, resume, transcript })); }
    catch (err) { setError(err instanceof Error ? err.message : "The analysis could not be completed. Please try again."); }
    finally { setLoading(false); }
  }

  function loadSample() {
    const s = getSampleFiles();
    setCandidateName(s.candidateName);
    setJobDescription(s.jobDescription);
    setResume(s.resume);
    setTranscript(s.transcript);
    setError("");
  }

  return <div className="intake-layout"><form onSubmit={submit} className="intake-form" aria-busy={loading}>
    <fieldset disabled={loading} className="form-fieldset">
      <section className="panel form-section"><div className="section-heading"><div className="section-title"><span className="step-number">01</span><div><h2>Candidate information</h2><p>Provide the candidate's name and the role they're interviewing for.</p></div></div><div className="section-meta" style={{ display: "flex", gap: "8px", alignItems: "center" }}><button type="button" onClick={loadSample} className="button secondary" style={{ fontSize: "12px", padding: "4px 10px", height: "auto" }} id="load-sample-btn">⚡ Load sample data</button><span>SMART ANALYSIS</span></div></div><div className="form-body"><label className="text-field">Candidate name <span className="required-mark">*</span><input required autoComplete="name" value={candidateName} onChange={e => setCandidateName(e.target.value)} placeholder="e.g. Priya Sharma"/></label><label className="text-field">Role / job description <span className="optional-label">Optional</span><textarea rows={3} value={jobDescription} onChange={e => setJobDescription(e.target.value)} placeholder="Describe the role, expected skills, and level of experience…"/><span className="field-hint">Used to compare the resume and answers with role expectations.</span></label></div></section>

      <section className="panel form-section"><div className="section-heading"><div className="section-title"><span className="step-number">02</span><div><h2>Upload resume &amp; transcript</h2><p>Just two files — we'll handle the rest automatically.</p></div></div><span className="document-count">{[resume, transcript].filter(Boolean).length} of 2 added</span></div><div className="form-body document-grid">
        <FileField label="Candidate resume" hint="PDF, DOCX, or TXT — the skills and experience claimed." file={resume} onChange={setResume} accept=".pdf,.docx,.txt"/>
        <FileField label="Interview transcript" hint="Free-form conversation text with Q&A from the interview." file={transcript} onChange={setTranscript} accept=".txt,.doc,.docx,.pdf"/>
        <div className="format-help"><Icon name="file" size={20}/><h3>Transcript format guide</h3><p>We auto-detect Q&A pairs from natural conversation transcripts.</p><button type="button" className="text-button" onClick={() => setFormatHelp(v => !v)} aria-expanded={formatHelp}>{formatHelp ? "Hide" : "View"} supported formats <Icon name="arrow" size={14}/></button></div>
      </div>
      {formatHelp && <div className="format-examples"><div><b>Format 1: Labelled speakers</b><pre>{"Interviewer: What is Python's GIL?\nCandidate: The GIL is a mutex that protects...\n\nInterviewer: Explain Docker containers.\nCandidate: Docker containers are lightweight..."}</pre></div><div><b>Format 2: Q&A style</b><pre>{"Q: What is a REST API?\nA: REST is an architectural style for...\n\nQ: Explain microservices.\nA: Microservices is a design pattern where..."}</pre></div><div><b>Format 3: Named speakers</b><pre>{"John: Tell me about your experience with AWS.\nPriya: I've worked with EC2, S3, and Lambda...\n\nJohn: How would you design a scalable system?\nPriya: I would start with load balancing..."}</pre></div></div>}
      </section>

      <section className="panel form-section smart-info-section"><div className="section-heading"><div className="section-title"><span className="step-number"><Icon name="shield" size={18}/></span><div><h2>What Smart Analysis does</h2><p>Powered by semantic matching and keyword intelligence.</p></div></div></div><div className="smart-features-grid">
        <div className="smart-feature"><span className="smart-feature-icon">🔍</span><strong>Auto-detects Q&A pairs</strong><p>Parses natural conversation into structured questions and answers.</p></div>
        <div className="smart-feature"><span className="smart-feature-icon">🎯</span><strong>Skill verification</strong><p>Cross-references resume claims with demonstrated knowledge in answers.</p></div>
        <div className="smart-feature"><span className="smart-feature-icon">📊</span><strong>Keyword intelligence</strong><p>Detects buzzword-stuffing, hollow claims, and unsubstantiated skills.</p></div>
        <div className="smart-feature"><span className="smart-feature-icon">🧠</span><strong>Semantic matching</strong><p>Uses TF-IDF similarity to compare answers with resume claims.</p></div>
      </div></section>
    </fieldset>
    {error && <div className="error-panel" role="alert"><div><strong>Analysis could not be completed.</strong><p>{error}</p></div></div>}
    <div className="submit-row"><div><Icon name="shield" size={18}/><span>Resume vs. transcript intelligence.<br/><strong>Does the candidate know what they claim?</strong></span></div><button type="submit" className="button primary" disabled={!canSubmit}>{loading ? <><span className="spinner small"/>Analyzing transcript & resume…</> : <>Analyze interview<Icon name="arrow" size={18}/></>}</button></div>
    {!canSubmit && !loading && <p className="submit-hint">Add a candidate name, resume, and transcript to continue.</p>}
  </form><aside className="intake-aside"><div className="review-preview"><div className="eyebrow">SMART ANALYSIS</div><h2>Two files.<br/><em>Complete insight.</em></h2><div className="preview-illustration" aria-hidden="true"><div className="mini-sheet"><div className="mini-sheet-head"><span/><i/></div><div className="mini-line long"/><div className="mini-line"/><div className="mini-chart"><i/><i/><i/><i/><i/></div><div className="mini-sheet-foot"><span/><span/></div></div><span className="illustration-stamp"><Icon name="check" size={23}/></span></div><ul className="benefit-list"><li><Icon name="check" size={16}/><div><strong>No formatting needed</strong><p>Upload a natural conversation transcript — we extract the structure.</p></div></li><li><Icon name="check" size={16}/><div><strong>Resume intelligence</strong><p>Detects buzzword-stuffing and hollow claims automatically.</p></div></li><li><Icon name="check" size={16}/><div><strong>Skill cross-reference</strong><p>Every resume skill checked against what was actually demonstrated.</p></div></li></ul></div></aside></div>;
}
