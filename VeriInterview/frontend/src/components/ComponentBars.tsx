import type { ModuleResult } from "../api";
const LABELS: Record<string, string> = { identity_risk: "Identity consistency", sync_risk: "Audio / video alignment", behavior_risk: "Gaze & surroundings", speech_behavior_risk: "Speech patterns", knowledge_ownership_risk: "Knowledge ownership", profile_consistency_risk: "Resume & answer consistency" };
export default function ComponentBars({ components }: { components: Record<string, ModuleResult> }) {
  return <div className="component-list">{Object.entries(LABELS).filter(([key]) => key in components).map(([key, label]) => {
    const mod = components[key]; const available = mod.status === "OK";
    return <details className="component-row" key={key}><summary><span className={`signal-dot ${available ? mod.level.toLowerCase() : "unavailable"}`}/><span className="component-name">{label}</span><span className="component-track"><span className={available ? mod.level.toLowerCase() : "unavailable"} style={{ width: available ? `${Math.max(2, mod.score)}%` : "100%" }}/></span><span className={`component-value ${available ? mod.level.toLowerCase() : "muted"}`}>{available ? `${Math.round(mod.score)} / 100` : "Not assessed"}</span><span className="detail-chevron">⌄</span></summary><div className="component-detail">{mod.evidence.map((e, i) => <p key={i}>{e.message}</p>)}{available && <small>Evidence confidence: {Math.round(mod.confidence * 100)}%</small>}</div></details>;
  })}</div>;
}
