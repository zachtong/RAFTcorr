export function arrowRenderUrl(
  idx: number,
  params: Record<string, string | number | boolean>
): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  return `/api/arrows/render/${idx}?${qs}`;
}
