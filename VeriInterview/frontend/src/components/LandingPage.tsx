import Icon from "./Icon";

interface Props {
  onStartReview: () => void;
  onSeeSample: () => void;
}

const SIGNALS: { icon: Parameters<typeof Icon>[0]["name"]; label: string }[] = [
  { icon: "file", label: "Resume consistency" },
  { icon: "shield", label: "Knowledge ownership" },
  { icon: "people", label: "Face identity" },
  { icon: "video", label: "Lip-sync" },
  { icon: "grid", label: "Behavior" },
  { icon: "help", label: "Speech" },
];

const FEATURES = [
  {
    title: "Resume claims, tested",
    body: "Every skill a candidate lists gets cross-referenced against what they actually demonstrate in the interview, not just whether they mentioned it.",
  },
  {
    title: "A counterfactual stress-test",
    body: "Change one assumption mid-answer. A candidate who understands the material adapts their reasoning. One who memorized a script doesn't.",
  },
  {
    title: "Evidence, not a verdict",
    body: "Every score comes with the exact evidence behind it, tagged as measured, inferred, or recommended, so a human makes the final call.",
  },
];

const STEPS = [
  { n: "01", title: "Upload", body: "A resume, the ideal-answer document, and a transcript, plus a video if you want the lip-sync check." },
  { n: "02", title: "Fuse", body: "Six signals combine with capped weights, so no single flag can dominate the score." },
  { n: "03", title: "Explain", body: "Every point of the score traces back to a specific, named piece of evidence." },
  { n: "04", title: "Decide", body: "You review the evidence. VeriInterview never makes the hiring call." },
];

export default function LandingPage({ onStartReview, onSeeSample }: Props) {
  return (
    <div className="landing">
      <header className="landing-topbar">
        <span className="brand">
          <span className="brand-mark"><Icon name="shield" size={22} /></span>
          <span>veri<span className="brand-light">interview</span></span>
        </span>
        <nav className="landing-topbar-actions">
          <button className="text-button" onClick={onSeeSample}>See a sample report</button>
          <button className="button primary" onClick={onStartReview}>Start a review</button>
        </nav>
      </header>

      <main>
        {/* 1 — Hero: stacked center, Mid Editorial scale, subtle textured surface */}
        <section className="landing-hero">
          <div className="landing-hero-inner">
            <span className="eyebrow">INTERVIEW INTEGRITY, MADE LEGIBLE</span>
            <h1>See what the resume doesn't say.</h1>
            <p className="landing-hero-sub">
              VeriInterview checks whether a candidate's claimed skills, spoken answers, and on-camera behavior
              actually agree, and shows you exactly why.
            </p>
            <div className="landing-hero-actions">
              <button className="button primary" onClick={onStartReview}>Start a review<Icon name="arrow" size={16} /></button>
              <button className="text-button" onClick={onSeeSample}>Open a sample report</button>
            </div>
            <p className="landing-hero-note">Never an automated hiring decision. Synthetic sample data included.</p>
          </div>
        </section>

        {/* 2 — Signals bar: centered row, flat color block */}
        <section className="landing-signals">
          <span className="eyebrow">SIX INDEPENDENT SIGNALS</span>
          <div className="landing-signals-row">
            {SIGNALS.map(s => (
              <div className="landing-signal" key={s.label}>
                <Icon name={s.icon} size={20} />
                <span>{s.label}</span>
              </div>
            ))}
          </div>
          <p className="landing-signals-caption">No single signal decides anything. They have to agree.</p>
        </section>

        {/* 3 — Features: gapless bento grid, solid surface */}
        <section className="landing-features">
          <div className="landing-section-heading">
            <span className="eyebrow">HOW THE EVIDENCE IS BUILT</span>
            <h2>Three ideas the score is built on.</h2>
          </div>
          <div className="landing-feature-grid">
            {FEATURES.map(f => (
              <div className="landing-feature-card" key={f.title}>
                <h3>{f.title}</h3>
                <p>{f.body}</p>
              </div>
            ))}
          </div>
        </section>

        {/* 4 — Product showcase: off-grid editorial, illustrative report preview */}
        <section className="landing-showcase">
          <div className="landing-showcase-copy">
            <span className="eyebrow">THE EXPLAINABLE BREAKDOWN</span>
            <h2>Open any report and see the reasoning, not just the number.</h2>
            <p>
              Every component carries its own weight, its own raw score, and the exact points it contributed,
              next to the evidence that produced it.
            </p>
          </div>
          <div className="landing-showcase-card" aria-hidden="true">
            <div className="landing-showcase-card-top">
              <span className="badge medium">MEDIUM</span>
              <span className="landing-showcase-score">44<small>/ 100</small></span>
            </div>
            {[
              { label: "Knowledge ownership", pct: 62, pts: "16 pts" },
              { label: "Voice modulation", pct: 54, pts: "14 pts" },
              { label: "Resume consistency", pct: 41, pts: "9 pts" },
              { label: "Face identity", pct: 18, pts: "3 pts" },
            ].map(row => (
              <div className="landing-showcase-row" key={row.label}>
                <span>{row.label}</span>
                <div className="component-track"><span style={{ width: `${row.pct}%` }} /></div>
                <span className="landing-showcase-pts">{row.pts}</span>
              </div>
            ))}
            <span className="landing-showcase-caption">Illustrative example</span>
          </div>
        </section>

        {/* 5 — How it works: vertical rhythm, numbered steps */}
        <section className="landing-steps">
          <div className="landing-section-heading">
            <span className="eyebrow">FROM UPLOAD TO DECISION</span>
            <h2>Four steps, nothing hidden.</h2>
          </div>
          <div className="landing-steps-row">
            {STEPS.map(s => (
              <div className="landing-step" key={s.n}>
                <span className="step-number">{s.n}</span>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
              </div>
            ))}
          </div>
        </section>

        {/* 6 — Closing CTA: stacked center, mini minimalist close */}
        <section className="landing-cta">
          <h2>Bring the evidence. Make the call.</h2>
          <p>Synthetic sample data included, try it in under a minute.</p>
          <button className="button primary" onClick={onStartReview}>Start a review<Icon name="arrow" size={16} /></button>
          <p className="landing-hero-note">Never an automated hiring decision.</p>
        </section>
      </main>

      <footer className="landing-footer">
        <span><Icon name="shield" size={14} /> Built for human review</span>
        <span>VeriInterview · TCS Tech Day</span>
      </footer>
    </div>
  );
}
