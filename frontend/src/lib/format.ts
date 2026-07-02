export function number(value: number | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function rate(value: number | undefined): string {
  return number(value, value && value < 100 ? 1 : 0);
}

export function percent(value: number | undefined): string {
  return `${number(value, 0)}%`;
}

// Compact relative time from a unix-ms timestamp (bigint from proto int64).
export function since(unixMs: bigint | number): string {
  const ms = typeof unixMs === "bigint" ? Number(unixMs) : unixMs;
  if (!ms) return "-";
  const secs = Math.max(0, (Date.now() - ms) / 1000);
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}
