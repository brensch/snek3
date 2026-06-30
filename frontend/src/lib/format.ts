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
