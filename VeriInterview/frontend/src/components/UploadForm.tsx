import { useRef, useState } from "react";
import { analyzeIntake, getSampleFile } from "../api";
import type { FullReport } from "../api";
import Icon from "./Icon";

function FileField({ label, hint, file, onChange, accept, optional = false }: { label: string; hint: string; file: File | null; onChange: (file: File | null) => void; accept: string; optional?: boolean }) {
  const ref = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  function select(file: File | undefined) {
    if (!file) return;
    const allowed = accept === "video/*" ? file.type.startsWith("video/") || /\.(mp4|mov|webm|avi|mkv)$/i.test(file.name) : accept.split(",").some(ext => file.name.toLowerCase().endsWith(ext));
    if (!allowed) { setError(`Please choose ${accept === "video/*" ? "a video file" : accept.replaceAll(",", ", ")} for this field.`); return; }
    setError(""); onChange(file);
  }
  return <div className="file-field"><div className="field-title">{label}<span>{optional ? "Optional" : "Required"}</span></div><p className="field-hint">{hint}</p><input ref={ref} className="sr-only" type="file" accept={accept} aria-label={label} onChange={e => select(e.target.files?.[0])} tabIndex={-1}/><div className={`dropzone ${file ? "has-file" : ""} ${dragging ? "dragging" : ""}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); select(e.dataTransfer.files[0]); }}>
    <button type="button" className="file-select" onClick={() => ref.current?.click()}><span className="file-icon"><Icon name={file ? "check" : optional ? "video" : "upload"} size={20}/></span><span className="file-description"><strong>{file ? file.name : "Click to upload or drop a file"}</strong><small>{file ? `${(file.size / 1024).toFixed(0)} KB · Click to replace` : optional ? "MP4, MOV, WebM" : accept.replaceAll(".", "").replaceAll(",", " / ").toUpperCase()}</small></span></button>{file && <button type="button" className="icon-button remove-file" aria-label={`Remove ${label}`} onClick={() => { onChange(null); if (ref.current) ref.current.value = ""; }}><Icon name="close" size={16}/></button>}
  </div>{error && <p className="field-error" role="alert">{error}</p>}</div>;
}

export default function UploadForm({ onResult, onDemo }: { onResult: (report: FullReport & { video_analysis_note: string }) => void; onDemo: () => void }) {
  const [candidateName, setCandidateName] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [resume, setResume] = useState<File | null>(null);
  const [idealAnswers, setIdealAnswers] = useState<File | null>(null);
  const [transcript, setTranscript] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [error, setError] = useState("");
  const [formats, setFormats] = useState(false);
  const ready = [resume, idealAnswers, transcript].filter(Boolean).length;
  const canSubmit = candidateName.trim() && ready === 3 && !loading && !sampleLoading;
  async function loadSample() {
    setSampleLoading(true); setError("");
    try {
      const files = await Promise.all([getSampleFile("resume_genuine_strong.txt"), getSampleFile("ideal_answers.txt"), getSampleFile("transcript_genuine_strong.txt")]);
      setCandidateName("Ananya Rao"); setJobDescription("Backend Engineer with Python, AWS (EC2, S3, Lambda, IAM), and machine learning experience.");
      setResume(files[0]); setIdealAnswers(files[1]); setTranscript(files[2]); setVideo(null);
    } catch (err) { setError(err instanceof Error ? err.message : "Sample files could not be loaded."); }
    finally { setSampleLoading(false); }
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault(); if (!canSubmit || !resume || !idealAnswers || !transcript) return;
    setLoading(true); setError("");
    try { onResult(await analyzeIntake({ candidateName: candidateName.trim(), jobDescription, resume, idealAnswers, transcript, video })); }
    catch (err) { setError(err instanceof Error ? err.message : "The review could not be completed. Please try again."); }
    finally { setLoading(false); }
  }
  return <div className="intake-layout"><form onSubmit={submit} className="intake-form" aria-busy={loading}>
    <fieldset disabled={loading || sampleLoading} className="form-fieldset">
      <section className="panel form-section"><div className="section-heading"><div className="section-title"><span className="step-number">01</span><div><h2>Set the context</h2><p>A little background makes the review more useful.</p></div></div><span className="section-meta">CANDIDATE DETAILS</span></div><div className="form-body"><label className="text-field">Candidate name <span className="required-mark">*</span><input required autoComplete="name" value={candidateName} onChange={e => setCandidateName(e.target.value)} placeholder="e.g. Ananya Rao"/></label><label className="text-field">Role / job description <span className="optional-label">Optional</span><textarea rows={3} value={jobDescription} onChange={e => setJobDescription(e.target.value)} placeholder="Describe the role, expected skills, and level of experience…"/><span className="field-hint">Used to compare the resume and answers with role expectations.</span></label></div></section>
      <section className="panel form-section"><div className="section-heading"><div className="section-title"><span className="step-number">02</span><div><h2>Add the interview documents</h2><p>The three essentials for a useful review.</p></div></div><span className="document-count">{ready} of 3 added</span></div><div className="form-body document-grid"><FileField label="Candidate resume" hint="The skills and experience the candidate has claimed." file={resume} onChange={setResume} accept=".pdf,.docx,.txt"/><FileField label="Reference answers" hint="Your questions, ideal answers, and required concepts." file={idealAnswers} onChange={setIdealAnswers} accept=".txt"/><FileField label="Interview transcript" hint="The candidate’s answers, response times, and durations." file={transcript} onChange={setTranscript} accept=".txt"/><div className="format-help"><Icon name="file" size={20}/><h3>A small note on formatting</h3><p>Reference answers and transcripts use a simple question-by-question template.</p><button type="button" className="text-button" onClick={() => setFormats(v => !v)} aria-expanded={formats}>{formats ? "Hide" : "View"} document format <Icon name="arrow" size={14}/></button></div></div>{formats && <div className="format-examples"><div><b>Reference answers</b><pre>{"[Q1]\nskill: Python\nquestion: What is a generator?\nreference_answer: A generator yields values lazily.\nconcepts: yield, lazy evaluation"}</pre><a href="/samples/ideal_answers.txt" download className="text-button">Download reference template <Icon name="download" size={14}/></a></div><div><b>Transcript</b><pre>{"[Q1]\nanswer: A generator uses yield to produce values.\nlatency: 2.0\nduration: 30.0"}</pre><a href="/samples/transcript_genuine_strong.txt" download className="text-button">Download transcript template <Icon name="download" size={14}/></a></div></div>}</section>
      <section className="panel form-section"><div className="section-heading"><div className="section-title"><span className="step-number">03</span><div><h2>Additional context</h2><p>Optional material for the prototype workflow.</p></div></div><span className="section-meta">OPTIONAL</span></div><div className="form-body video-grid"><FileField label="Interview video" hint="Attach the recording, if available." file={video} onChange={setVideo} accept="video/*" optional/><div className="prototype-note"><span className="eyebrow">VIDEO CHECKS ARE SIMULATED</span><p>This build does not analyze faces or lip-sync in uploaded video. Video-derived scores are demonstration placeholders and should not guide a candidate assessment.</p></div></div></section>
    </fieldset>
    {error && <div className="error-panel" role="alert"><div><strong>We couldn’t complete the review.</strong><p>{error}</p></div></div>}
    <div className="submit-row"><div><Icon name="shield" size={18}/><span>For interviewer review.<br/><strong>You make the final decision.</strong></span></div><button type="submit" className="button primary" disabled={!canSubmit}>{loading ? <><span className="spinner small"/>Preparing your review…</> : <>Review interview<Icon name="arrow" size={18}/></>}</button></div>
    {!canSubmit && !loading && <p className="submit-hint">Add a candidate name and all three documents to continue.</p>}
  </form><aside className="intake-aside"><div className="review-preview"><div className="eyebrow">WHAT YOU’LL GET</div><h2>The full picture,<br/><em>with context.</em></h2><div className="preview-illustration" aria-hidden="true"><div className="mini-sheet"><div className="mini-sheet-head"><span/><i/></div><div className="mini-line long"/><div className="mini-line"/><div className="mini-chart"><i/><i/><i/><i/><i/></div><div className="mini-sheet-foot"><span/><span/></div></div><span className="illustration-stamp"><Icon name="check" size={23}/></span></div><ul className="benefit-list"><li><Icon name="check" size={16}/><div><strong>Skills, put to the test</strong><p>Compare resume claims with demonstrated knowledge.</p></div></li><li><Icon name="check" size={16}/><div><strong>Evidence you can follow</strong><p>See the answers and signals behind the assessment.</p></div></li><li><Icon name="check" size={16}/><div><strong>A useful next step</strong><p>Know where a follow-up conversation may help.</p></div></li></ul></div><div className="sample-card"><span className="eyebrow">FIRST TIME HERE?</span><h3>Take a look around.</h3><p>Use a fictional candidate’s documents to try the review workflow.</p><button type="button" className="button secondary" onClick={loadSample} disabled={loading || sampleLoading}>{sampleLoading ? "Loading documents…" : "Use sample documents"}<Icon name="arrow" size={15}/></button><button type="button" className="text-button" onClick={onDemo}>View completed interviews <Icon name="arrow" size={14}/></button></div></aside></div>;
}
