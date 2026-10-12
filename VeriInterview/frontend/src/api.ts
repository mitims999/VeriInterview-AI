export interface Evidence {
  type: string;
  severity: "low" | "medium" | "high";
  message: string;
  question_id?: string | null;
  timestamp?: number | null;
}

export interface ModuleResult {
  feature: string;
  score: number;
  confidence: number;
  level: "LOW" | "MEDIUM" | "HIGH" | "UNAVAILABLE";
  evidence: Evidence[];
  quality: { data_quality: number; baseline_sufficient: boolean };
  status: "OK" | "INSUFFICIENT_EVIDENCE" | "UNAVAILABLE";
}

export interface RiskAssessment {
  candidate_name: string;
  risk_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  confidence: number;
  capability_score: number;
  components: Record<string, ModuleResult>;
  evidence_timeline: Evidence[];
  recommended_action: string;
}

export interface SkillVerdict {
  skill: string;
  verdict: string;
  evidence_score: number;
  questions_used: string[];
}

export interface AnswerEvaluation {
  question_id: string;
  score: number;
  concept_coverage: number;
  missing_concepts: string[];
  incorrect_concepts: string[];
  confidence: number;
}

export interface QAExchange {
  question_id: string;
  question_text: string;
  skill: string;
  difficulty: string;
  expected_proficiency: string;
  reference_answer: string;
  required_concepts: string[];
  candidate_answer: string;
  response_latency_sec: number;
  answer_duration_sec: number;
}

export interface FullReport {
  assessment: RiskAssessment;
  skill_profile: {
    skill_match_score: number;
    matched_keywords: string[];
    missing_keywords: string[];
    resume_claims: { skill: string; claim_text: string; claim_strength: string }[];
  };
  skill_verdicts: SkillVerdict[];
  answer_evaluations: AnswerEvaluation[];
  buzzword_coverage: {
    buzzword_coverage_score: number;
    expected_keywords: string[];
    used_keywords: string[];
    missing_keywords: string[];
  };
  qa_list: QAExchange[];
  scenario_label?: string;
  expected_outcome?: string;
}

export interface DemoCandidate {
  id: string;
  candidate_name: string;
  label: string;
}

const BASE = "/api";

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status}). Please check the documents and try again.`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch { /* A proxy may return HTML when the API is unavailable. */ }
    throw new Error(message);
  }
  try { return await response.json() as T; }
  catch { throw new Error("The analysis service is unavailable. Start the backend and try again."); }
}

export async function getSampleFile(name: string): Promise<File> {
  const response = await fetch(`/samples/${encodeURIComponent(name)}`);
  if (!response.ok || response.headers.get("content-type")?.includes("text/html")) {
    throw new Error("Sample documents are unavailable. Make sure the backend is running on port 8000.");
  }
  return new File([await response.blob()], name, { type: "text/plain" });
}

export async function listDemoCandidates(): Promise<DemoCandidate[]> {
  const r = await fetch(`${BASE}/demo/candidates`);
  return readResponse(r);
}

export async function getDemoReport(id: string): Promise<FullReport> {
  const r = await fetch(`${BASE}/demo/candidates/${id}/report`);
  return readResponse(r);
}

export async function listSamples(): Promise<string[]> {
  const r = await fetch(`${BASE}/samples`);
  return readResponse(r);
}

export interface AnalyzeInput {
  candidateName: string;
  jobDescription: string;
  resume: File;
  idealAnswers: File;
  transcript: File;
  video?: File | null;
}

export async function analyzeIntake(input: AnalyzeInput): Promise<FullReport & { video_analysis_note: string }> {
  const fd = new FormData();
  fd.append("candidate_name", input.candidateName);
  fd.append("job_description", input.jobDescription);
  fd.append("resume", input.resume);
  fd.append("ideal_answers", input.idealAnswers);
  fd.append("transcript", input.transcript);
  if (input.video) fd.append("video", input.video);
  const r = await fetch(`${BASE}/analyze`, { method: "POST", body: fd });
  return readResponse(r);
}

export interface TestedSkill {
  skill: string;
  in_resume: boolean;
  tested_in_interview: boolean;
  questions_used: string[];
  demonstrated_score: number;
  semantic_similarity: number;
  support_level: "STRONG" | "MODERATE" | "SURFACE" | "MISMATCH" | string;
  match_status: "MATCH" | "PARTIAL" | "SURFACE" | "MISMATCH" | string;
  message: string;
}

export interface UntestedSkill {
  skill: string;
  in_resume: boolean;
  tested_in_interview: boolean;
  support_level: "NOT_TESTED" | string;
  match_status: "NOT_TESTED" | string;
  semantic_similarity: number;
  message: string;
}

export interface SkillCrossReference {
  skill: string;
  in_resume?: boolean;
  tested_in_interview?: boolean;
  questions_used?: string[];
  demonstrated_score?: number;
  mentioned_in_transcript?: boolean;
  semantic_similarity: number;
  support_level: string;
  match_status?: string;
  message: string;
}

export interface KeywordDetection {
  hollow_claims: { phrase: string; type: string; severity: string; message: string }[];
  buzzword_analysis: {
    total_buzzwords: number;
    buzzwords_found: string[];
    substantiated: string[];
    unsubstantiated: string[];
    density_percent: number;
    is_stuffed: boolean;
  };
  extracted_skills: string[];
  cross_reference: {
    tested_skills?: TestedSkill[];
    untested_skills?: UntestedSkill[];
    skill_cross_reference: SkillCrossReference[];
    total_skills: number;
    tested_count?: number;
    verified_count?: number;
    mismatch_count?: number;
    untested_count?: number;
    supported_count: number;
    unsupported_count: number;
    surface_only_count?: number;
    match_rate_percent?: number;
    support_ratio: number;
  } | null;
  resume_suspicion_score: number;
  summary: string;
}

export interface SmartReport extends FullReport {
  keyword_detection: KeywordDetection;
  transcript_skills: string[];
  parsed_questions_count: number;
  analysis_mode: "smart";
}

export interface SmartAnalyzeInput {
  candidateName: string;
  jobDescription: string;
  resume: File;
  transcript: File;
}

export async function smartAnalyze(input: SmartAnalyzeInput): Promise<SmartReport> {
  const fd = new FormData();
  fd.append("candidate_name", input.candidateName);
  fd.append("job_description", input.jobDescription);
  fd.append("resume", input.resume);
  fd.append("transcript", input.transcript);
  const r = await fetch(`${BASE}/smart-analyze`, { method: "POST", body: fd });
  return readResponse(r);
}

