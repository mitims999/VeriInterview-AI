import { useState } from "react";
import type { SmartReport } from "../api";
import RiskGauge from "./RiskGauge";
import ComponentBars from "./ComponentBars";
import SkillTable from "./SkillTable";
import QuestionCard from "./QuestionCard";
import EvidenceTimeline from "./EvidenceTimeline";
import KeywordReport from "./KeywordReport";
import Icon from "./Icon";

type Tab = "overview" | "answers" | "keywords" | "evidence";

export default function SmartReportView({ report }: { report: SmartReport }) {
  const [tab, setTab] = useState<Tab>("overview");
  const assessment = report.assessment;
  const evalByQ = new Map(report.answer_evaluations.map(e => [e.question_id, e]));
  const available = Object.values(assessment.components).filter(c => c.status === "OK").length;
  const kw = report.keyword_detection;

  function exportReport() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = `${assessment.candidate_name.replace(/[^a-z0-9]/gi, "-")}-smart-review.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <div className="report">
    <div className="report-context">
      <span className="report-meta"><Icon name="file" size={16}/>{report.parsed_questions_count} questions auto-detected<span>·</span>{kw.cross_reference?.tested_count ?? report.skill_verdicts.length} skills tested<span>·</span>{kw.extracted_skills.length} resume skills scanned</span>
      <button className="text-button" onClick={exportReport}><Icon name="download" size={16}/>Export report</button>
    </div>

    <div className="notice smart-notice"><strong>Smart Analysis</strong><span>Q&A pairs were automatically extracted from the transcript. {report.parsed_questions_count} questions detected. Evaluated against resume claims and candidate answers.</span></div>

    <div className="report-summary">
      <section className="panel"><RiskGauge score={assessment.risk_score} level={assessment.risk_level} confidence={assessment.confidence}/></section>
      <section className="panel capability-summary">
        <div className="eyebrow">DEMONSTRATED CAPABILITY</div>
        <div className="capability-number">{Math.round(assessment.capability_score)}<span>/ 100</span></div>
        <p>Technical ability, assessed separately.</p>
        <div className="capability-track"><span style={{ width: `${assessment.capability_score}%` }}/></div>
        <p className="metric-note">Based on answer quality and coverage of role-specific concepts. Lower ability does not establish an integrity concern.</p>
      </section>
      <section className="panel keyword-summary-panel">
        <div className="eyebrow">RESUME INTELLIGENCE</div>
        <div className="capability-number">{Math.round(kw.resume_suspicion_score)}<span>/ 100</span></div>
        <p>Resume suspicion score.</p>
        <div className="capability-track"><span style={{ width: `${kw.resume_suspicion_score}%`, background: kw.resume_suspicion_score > 50 ? 'var(--red, #ef4444)' : kw.resume_suspicion_score > 20 ? 'var(--amber, #f59e0b)' : 'var(--green, #22c55e)' }}/></div>
        <p className="metric-note">{kw.cross_reference ? `${kw.cross_reference.verified_count ?? kw.cross_reference.supported_count} of ${kw.cross_reference.tested_count ?? kw.cross_reference.total_skills} tested skills verified (${kw.cross_reference.match_rate_percent ?? 100}% match).` : 'Resume keyword analysis complete.'} {kw.cross_reference?.mismatch_count ? `⚠ ${kw.cross_reference.mismatch_count} mismatch(es). ` : ''} {kw.cross_reference?.untested_count ? `${kw.cross_reference.untested_count} skills not tested.` : ''}</p>
      </section>
    </div>

    <section className="recommendation"><span className="recommendation-icon"><Icon name="shield" size={22}/></span><div className="eyebrow">SUGGESTED NEXT STEP</div><h2>{assessment.confidence < 0.5 ? "Gather more context." : assessment.risk_level === "LOW" ? "Continue with your review." : "Have a closer conversation."}</h2><p>{assessment.recommended_action}</p><span className="human-note">A recommendation, never a hiring decision.</span></section>

    <div className="report-tabs" role="tablist" aria-label="Report sections">
      {([['overview', 'Overview'], ['answers', `Answers (${report.qa_list.length})`], ['keywords', 'Resume Intelligence'], ['evidence', 'Evidence log']] as const).map(([id, label]) =>
        <button key={id} id={`tab-${id}`} role="tab" aria-selected={tab === id} aria-controls={`panel-${id}`} onClick={() => setTab(id)} className={`${tab === id ? "selected" : ""} ${id === "keywords" ? "keyword-tab" : ""}`}>{label}{id === "keywords" && kw.resume_suspicion_score > 40 && <span className="tab-alert">!</span>}</button>
      )}
    </div>

    <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
      {tab === "overview" && <div className="overview-panels">
        <section className="panel"><div className="section-heading"><div><h2>The evidence at a glance</h2><p>Open a signal to see what contributed to its score.</p></div><span className="section-meta">HIGHER = MORE RISK</span></div><ComponentBars components={assessment.components}/></section>
        <section className="panel"><div className="section-heading"><div><h2>Do the skills hold up?</h2><p>Resume claims compared with the interview answers.</p></div><Icon name="grid" size={19}/></div><SkillTable verdicts={report.skill_verdicts}/><div className="table-footer"><span>Resume / role match <strong>{Math.round(report.skill_profile.skill_match_score)}%</strong></span><span>Role keyword coverage <strong>{Math.round(report.buzzword_coverage.buzzword_coverage_score)}%</strong></span></div></section>
      </div>}

      {tab === "answers" && <section className="panel"><div className="section-heading"><div><h2>Inside the conversation</h2><p>Auto-extracted Q&A pairs from the transcript, matched with resume skills.</p></div></div>{report.qa_list.length ? <div className="question-list">{report.qa_list.map(qa => <QuestionCard key={qa.question_id} qa={qa} evalu={evalByQ.get(qa.question_id)}/>)}</div> : <p className="empty-state">No questions were available for review.</p>}</section>}

      {tab === "keywords" && <section className="panel"><div className="section-heading"><div><h2>Resume Intelligence &amp; Keywords</h2><p>Buzzword detection, hollow claims, and skill cross-referencing.</p></div></div><KeywordReport data={kw}/></section>}

      {tab === "evidence" && <section className="panel"><div className="section-heading"><div><h2>Every signal, in context</h2><p>Trace observations back to the assessment. Flags invite review; they do not establish dishonesty.</p></div></div><EvidenceTimeline evidence={assessment.evidence_timeline}/></section>}
    </div>

    <div className="report-disclaimer"><Icon name="shield" size={17}/><p>This assessment supports human review. Consider missing evidence, recording quality, and the candidate's circumstances before drawing conclusions.</p></div>
  </div>;
}
