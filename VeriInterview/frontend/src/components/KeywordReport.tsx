import type { KeywordDetection } from "../api";
import Icon from "./Icon";

const STATUS_COLORS: Record<string, string> = {
  STRONG: "var(--green, #22c55e)",
  MODERATE: "var(--teal, #14b8a6)",
  SURFACE: "var(--amber, #f59e0b)",
  MISMATCH: "var(--red, #ef4444)",
  NOT_TESTED: "var(--muted, #94a3b8)",
};

const STATUS_LABELS: Record<string, string> = {
  STRONG: "Verified Match",
  MODERATE: "Moderate Match",
  SURFACE: "Surface Only",
  MISMATCH: "Skill Mismatch",
  NOT_TESTED: "Not Tested",
};

export default function KeywordReport({ data }: { data: KeywordDetection }) {
  const crossRef = data.cross_reference;
  const testedSkills = crossRef?.tested_skills || [];
  const untestedSkills = crossRef?.untested_skills || [];
  const testedCount = crossRef?.tested_count ?? testedSkills.length;
  const verifiedCount = crossRef?.verified_count ?? 0;
  const mismatchCount = crossRef?.mismatch_count ?? 0;
  const matchRate = crossRef?.match_rate_percent ?? (testedCount > 0 ? Math.round((verifiedCount / testedCount) * 100) : 100);

  return (
    <div className="keyword-report">
      {/* Summary banner */}
      <div
        className={`keyword-summary-banner ${
          data.resume_suspicion_score > 50
            ? "high"
            : data.resume_suspicion_score > 20
            ? "medium"
            : "low"
        }`}
      >
        <div className="keyword-summary-score">
          <span className="keyword-score-number">
            {Math.round(data.resume_suspicion_score)}
          </span>
          <span className="keyword-score-label">
            Resume
            <br />
            Suspicion
          </span>
        </div>
        <div>
          <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "6px" }}>
            <span
              style={{
                fontSize: "13px",
                fontWeight: 600,
                padding: "2px 8px",
                borderRadius: "4px",
                background: mismatchCount > 0 ? "rgba(239,68,68,0.2)" : "rgba(34,197,94,0.2)",
                color: mismatchCount > 0 ? "var(--red, #ef4444)" : "var(--green, #22c55e)",
              }}
            >
              {mismatchCount > 0 ? `⚠ ${mismatchCount} Skill Mismatch(es)` : `✓ ${matchRate}% Tested Skill Match`}
            </span>
            <span style={{ fontSize: "13px", color: "var(--muted, #64748b)" }}>
              {verifiedCount} of {testedCount} tested skills verified
            </span>
          </div>
          <p className="keyword-summary-text">{data.summary}</p>
        </div>
      </div>

      {/* Tested Skills Section — The Core Verification */}
      <div className="keyword-section">
        <div className="section-heading">
          <div>
            <h3>Skills tested in interview ({testedSkills.length})</h3>
            <p>
              Resume skills that were actually asked about and demonstrated in candidate answers.
            </p>
          </div>
          <span className="section-meta" style={{ color: mismatchCount > 0 ? "var(--red, #ef4444)" : "inherit" }}>
            {verifiedCount} of {testedCount} verified ({matchRate}% match)
          </span>
        </div>

        {testedSkills.length > 0 ? (
          <div className="skill-cross-ref-grid">
            {testedSkills.map((skill) => {
              const color = STATUS_COLORS[skill.support_level] || "#6b7280";
              const label = STATUS_LABELS[skill.support_level] || skill.support_level;
              return (
                <div
                  key={skill.skill}
                  className="skill-cross-ref-card"
                  style={{
                    borderColor: skill.support_level === "MISMATCH" ? "var(--red, #ef4444)" : undefined,
                    borderWidth: skill.support_level === "MISMATCH" ? "2px" : "1px",
                  }}
                >
                  <div className="skill-cross-ref-header">
                    <strong>{skill.skill.toUpperCase()}</strong>
                    <span className="skill-support-badge" style={{ background: color, color: "#fff" }}>
                      {label}
                    </span>
                  </div>
                  <p className="skill-cross-ref-message">{skill.message}</p>
                  <div className="skill-sim-bar">
                    <div
                      className="skill-sim-fill"
                      style={{
                        width: `${Math.min(100, Math.max(8, skill.demonstrated_score || skill.semantic_similarity * 100))}%`,
                        background: color,
                      }}
                    />
                  </div>
                  <div className="skill-sim-meta">
                    <span>Performance score: {Math.round(skill.demonstrated_score || 0)}%</span>
                    {skill.questions_used && skill.questions_used.length > 0 && (
                      <span className="skill-mentioned-badge">
                        Evaluated in {skill.questions_used.join(", ")}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="empty-state">No specific resume skills were tested in this interview.</p>
        )}
      </div>

      {/* Untested Skills Section — Neutral Context */}
      {untestedSkills.length > 0 && (
        <div className="keyword-section">
          <div className="section-heading">
            <div>
              <h3>Resume skills not tested in this interview ({untestedSkills.length})</h3>
              <p>
                These skills are listed on the resume but were not asked about. They do not penalize the candidate.
              </p>
            </div>
            <span className="section-meta">Neutral (Untested)</span>
          </div>
          <div className="buzzword-tags">
            {untestedSkills.map((s) => (
              <span
                key={s.skill}
                className="buzzword-tag neutral"
                title="Not discussed in this interview"
                style={{ opacity: 0.85 }}
              >
                {s.skill}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Buzzword Analysis */}
      <div className="keyword-section">
        <div className="section-heading">
          <div>
            <h3>Buzzword analysis</h3>
            <p>Technical keywords and their substantiation in the resume.</p>
          </div>
          {data.buzzword_analysis.is_stuffed && (
            <span className="stuffed-badge">⚠ Potential stuffing</span>
          )}
        </div>
        {data.buzzword_analysis.total_buzzwords > 0 ? (
          <div className="buzzword-grid">
            <div className="buzzword-stat">
              <span className="buzzword-stat-number">{data.buzzword_analysis.total_buzzwords}</span>
              <span>Buzzwords found</span>
            </div>
            <div className="buzzword-stat">
              <span className="buzzword-stat-number good">
                {data.buzzword_analysis.substantiated.length}
              </span>
              <span>Substantiated</span>
            </div>
            <div className="buzzword-stat">
              <span className="buzzword-stat-number warning">
                {data.buzzword_analysis.unsubstantiated.length}
              </span>
              <span>Unsubstantiated</span>
            </div>
            <div className="buzzword-stat">
              <span className="buzzword-stat-number">{data.buzzword_analysis.density_percent}%</span>
              <span>Density</span>
            </div>
            {data.buzzword_analysis.unsubstantiated.length > 0 && (
              <div className="buzzword-list">
                <strong>Unsubstantiated keywords:</strong>
                <div className="buzzword-tags">
                  {data.buzzword_analysis.unsubstantiated.map((bw) => (
                    <span key={bw} className="buzzword-tag warning">
                      {bw}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {data.buzzword_analysis.substantiated.length > 0 && (
              <div className="buzzword-list">
                <strong>Substantiated keywords:</strong>
                <div className="buzzword-tags">
                  {data.buzzword_analysis.substantiated.map((bw) => (
                    <span key={bw} className="buzzword-tag good">
                      {bw}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="empty-state">No trending buzzwords detected in the resume.</p>
        )}
      </div>

      {/* Hollow Claims */}
      {data.hollow_claims.length > 0 && (
        <div className="keyword-section">
          <div className="section-heading">
            <div>
              <h3>Hollow claims detected</h3>
              <p>Vague self-descriptors that add no verifiable signal.</p>
            </div>
            <span className="section-meta">{data.hollow_claims.length} found</span>
          </div>
          <div className="hollow-claims-list">
            {data.hollow_claims.map((claim) => (
              <div key={claim.phrase} className="hollow-claim-item">
                <Icon name="alert" size={16} />
                <span className="hollow-phrase">"{claim.phrase}"</span>
                <span className="hollow-message">{claim.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
