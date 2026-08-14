/**
 * Builds an `Authorization` header value from a user-supplied token that may
 * be a bare token or may already include the `Bearer ` prefix (the Settings
 * panel doesn't force a shape, so accept either rather than risk sending a
 * double-prefixed `Bearer Bearer <token>` header).
 */
export function authHeaderValue(rawToken: string): string {
  const trimmed = rawToken.trim();
  return /^bearer\s+/i.test(trimmed) ? trimmed : `Bearer ${trimmed}`;
}
