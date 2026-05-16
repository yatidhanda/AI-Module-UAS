from __future__ import annotations


CONSOLE_CSS = """
<style>
:root {
  --bg: #07111f;
  --panel: #0d1726;
  --panel-2: #101d2e;
  --border: rgba(148, 163, 184, 0.22);
  --text: #e5edf7;
  --muted: #94a3b8;
  --green: #4ade80;
  --amber: #f59e0b;
  --orange: #fb923c;
  --red: #f87171;
  --blue: #60a5fa;
  --gray: #9ca3af;
}

.stApp {
  background: radial-gradient(circle at 20% 0%, rgba(29, 78, 216, 0.12), transparent 28%),
              linear-gradient(180deg, #06101d 0%, #091320 42%, #07111f 100%);
  color: var(--text);
}

section[data-testid="stSidebar"] {
  background: #07111f;
  border-right: 1px solid var(--border);
}

.block-container {
  padding-top: 1.1rem;
  padding-bottom: 2rem;
  max-width: 1800px;
}

h1, h2, h3 {
  letter-spacing: 0;
}

div[data-testid="stMetric"] {
  background: rgba(13, 23, 38, 0.92);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.8rem 0.9rem;
  min-height: 94px;
}

div[data-testid="stMetricLabel"] {
  color: var(--muted);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

div[data-testid="stMetricValue"] {
  color: var(--text);
  font-size: 1.36rem;
}

.isr-header {
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(13, 23, 38, 0.78));
  border-radius: 8px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.8rem;
}

.isr-title {
  margin: 0;
  font-size: 1.65rem;
  font-weight: 700;
}

.isr-subtitle {
  color: var(--muted);
  margin-top: 0.15rem;
  font-size: 0.95rem;
}

.isr-meta {
  color: #cbd5e1;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.78rem;
  margin-top: 0.55rem;
}

.panel {
  border: 1px solid var(--border);
  background: rgba(13, 23, 38, 0.88);
  border-radius: 8px;
  padding: 0.9rem;
  margin-bottom: 0.8rem;
}

.panel-title {
  color: #dbeafe;
  font-size: 0.86rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.55rem;
}

.mission-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.55rem;
}

.mission-item {
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 7px;
  padding: 0.55rem 0.65rem;
}

.mission-label {
  color: var(--muted);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.mission-value {
  color: var(--text);
  font-size: 0.86rem;
  margin-top: 0.16rem;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.35rem 0 0.85rem;
}

.badge {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.24rem 0.55rem;
  font-size: 0.77rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.badge-green { color: #bbf7d0; border-color: rgba(74, 222, 128, 0.45); background: rgba(34, 197, 94, 0.12); }
.badge-amber { color: #fde68a; border-color: rgba(245, 158, 11, 0.55); background: rgba(245, 158, 11, 0.12); }
.badge-orange { color: #fed7aa; border-color: rgba(251, 146, 60, 0.6); background: rgba(251, 146, 60, 0.14); }
.badge-red { color: #fecaca; border-color: rgba(248, 113, 113, 0.55); background: rgba(248, 113, 113, 0.13); }
.badge-gray { color: #d1d5db; border-color: rgba(156, 163, 175, 0.4); background: rgba(156, 163, 175, 0.1); }
.badge-blue { color: #bfdbfe; border-color: rgba(96, 165, 250, 0.48); background: rgba(96, 165, 250, 0.1); }

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.small-muted {
  color: var(--muted);
  font-size: 0.82rem;
}

.section-callout {
  border-left: 3px solid var(--blue);
  background: rgba(96, 165, 250, 0.08);
  padding: 0.65rem 0.75rem;
  border-radius: 6px;
  color: #dbeafe;
}

.image-card {
  border: 1px solid var(--border);
  background: rgba(13, 23, 38, 0.9);
  border-radius: 8px;
  padding: 0.65rem;
  min-height: 100%;
}

.image-card-title {
  color: #dbeafe;
  font-size: 0.82rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 0.35rem;
}

.image-meta {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.35rem 0.65rem;
  margin-top: 0.55rem;
  font-size: 0.78rem;
}

.image-meta-label {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.66rem;
}

.image-meta-value {
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.action-console {
  border: 1px solid rgba(96, 165, 250, 0.34);
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.72));
  border-radius: 8px;
  padding: 0.9rem;
  margin: 0.85rem 0;
}

.track-preview-card {
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: rgba(7, 17, 31, 0.72);
  border-radius: 8px;
  padding: 0.65rem;
}

.threat-alert-card {
  border: 1px solid rgba(251, 146, 60, 0.4);
  background: rgba(15, 23, 42, 0.92);
  border-radius: 8px;
  padding: 0.75rem;
  margin-bottom: 0.55rem;
}

.threat-alert-card.critical {
  border-color: rgba(248, 113, 113, 0.6);
  background: rgba(127, 29, 29, 0.18);
}

.threat-alert-title {
  color: #e5edf7;
  font-size: 0.9rem;
  font-weight: 700;
}

.threat-alert-meta {
  color: var(--muted);
  font-size: 0.78rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  margin: 0.25rem 0;
}

.threat-note {
  color: var(--muted);
  font-size: 0.78rem;
  margin-top: 0.35rem;
}

.human-review-pill {
  display: inline-flex;
  color: #fde68a;
  border: 1px solid rgba(245, 158, 11, 0.6);
  background: rgba(245, 158, 11, 0.12);
  border-radius: 999px;
  padding: 0.35rem 0.7rem;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.04em;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 8px;
}

code {
  color: #dbeafe;
}
</style>
"""


def confidence_badge_class(level: str | None) -> str:
    level = (level or "unknown").lower()
    if level == "high":
        return "badge-green"
    if level == "medium":
        return "badge-amber"
    if level == "low":
        return "badge-red"
    return "badge-gray"


def alert_badge_class(level: str | None) -> str:
    level = (level or "gray").lower()
    if level == "green":
        return "badge-green"
    if level == "amber":
        return "badge-amber"
    if level == "red":
        return "badge-red"
    if level == "orange":
        return "badge-orange"
    if level == "blue":
        return "badge-blue"
    return "badge-gray"


def threat_badge_class(level: str | None) -> str:
    level = (level or "unknown").lower()
    if level == "critical":
        return "badge-red"
    if level == "high":
        return "badge-orange"
    if level == "medium":
        return "badge-amber"
    if level == "low":
        return "badge-green"
    return "badge-gray"
