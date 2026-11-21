/** Browser-side endpoint validation. A Vercel deployment must not accidentally call the visitor's localhost. */
export function normalizeBackendUrl(input: string): string {
  const trimmed = input.trim();
  if (!trimmed) return "";
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new Error("Introdu o adresă completă, de exemplu https://backend.example.com.");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Adresa backend-ului trebuie să înceapă cu http:// sau https://.");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error("Folosește numai adresa serverului, fără parolă, parametri sau fragment.");
  }
  if (url.pathname !== "/" && url.pathname !== "") {
    throw new Error("Folosește adresa de bază a serverului, fără /api sau /ws.");
  }
  return url.origin;
}

export function backendWebSocketUrl(backendUrl: string): string {
  const url = new URL(backendUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/ws/live";
  return url.toString();
}

export function backendConfigError(backendUrl: string, frontendProtocol: string): string | null {
  if (!backendUrl) return "Configurează URL-ul backend-ului Python (serverul nu rulează pe Vercel).";
  if (frontendProtocol === "https:" && backendUrl.startsWith("http:")) {
    return "Site-ul este HTTPS; backend-ul trebuie să fie tot HTTPS, cu WebSocket WSS.";
  }
  return null;
}
