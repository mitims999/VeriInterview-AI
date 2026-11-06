import type { XAIExplanation } from "../api";

const TAG_STYLE: Record<string, { label: string; color: string }> = {
  measured: { label: "Measured", color: "#38bdf8" },
  inference: { label: "Model inference", color: "#f59e0b" },
  insufficient_evidence: { label: "Insufficient evidence", color: "#6b7280" },
};

const LEVEL_COLOR: Record<string, string> = {
  LOW: "#22c55e",
  MEDIUM: "#eab308",
  HIGH: "#ef4444",
  UNAVAILABLE: "#4b5563",
};

export default function XAIPanel({ explanation }: { explanation: XAIExplanation }) {
  return (
    <section className="rounded-xl border border-[#242938] bg-[#11151d] p-6 space-y-5">
      <div>
        <h2 className="text-gray-200 font-semibold mb-1">Why this score — explainable breakdown</h2>
        <p className="text-sm text-gray-300">{explanation.narrative}</p>
      </div>

      <div className="space-y-3">
        {explanation.components.map((c) => {
          const tag = TAG_STYLE[c.tag];
          const levelColor = LEVEL_COLOR[c.level];
          return (
            <div key={c.key} className="rounded-lg border border-[#1f2430] bg-[#0e1118] p-3">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-gray-200 text-sm font-medium">{c.label}</span>
                  <span
                    className="rounded-full px-2 py-0.5 text-[10px] font-medium"
                    style={{ background: `${tag.color}22`, color: tag.color }}
                  >
                    {tag.label}
                  </span>
                </div>
                <div className="text-xs text-gray-400">
                  weight {Math.round(c.weight * 100)}% · raw {c.raw_score.toFixed(0)} ·{" "}
                  <span style={{ color: levelColor }} className="font-semibold">
                    +{c.contribution_points.toFixed(1)} pts
                  </span>
                </div>
              </div>
              <ul className="mt-2 space-y-1">
                {c.top_evidence.map((msg, i) => (
                  <li key={i} className="text-xs text-gray-400">
                    · {msg}
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>

      <div className="rounded-lg border border-[#1f2430] bg-[#0e1118] p-3">
        <div className="flex items-center justify-between">
          <span className="text-gray-200 text-sm font-medium">Authenticity (Face Identity + Lip-Sync + Blink/Mouth cues)</span>
          <span
            className="rounded-full px-2 py-0.5 text-[10px] font-medium"
            style={{
              background: `${LEVEL_COLOR[explanation.authenticity.level]}22`,
              color: LEVEL_COLOR[explanation.authenticity.level],
            }}
          >
            {explanation.authenticity.status === "OK" ? explanation.authenticity.level : "UNAVAILABLE"}
          </span>
        </div>
        <p className="mt-2 text-xs text-gray-400">{explanation.authenticity.summary}</p>
        <p className="mt-1 text-[11px] text-gray-600">
          Shown separately — identity and lip-sync evidence already count individually above, so this combined
          score is not added again into the risk total (avoids double-counting).
        </p>
      </div>
    </section>
  );
}
