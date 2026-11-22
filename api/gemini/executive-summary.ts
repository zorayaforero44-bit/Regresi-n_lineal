import type { IncomingMessage, ServerResponse } from "http";
import { generateExecutiveNarrative } from "../../server/geminiService.js";

async function readJsonBody(req: IncomingMessage): Promise<any> {
  if ((req as any).body) return (req as any).body;
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

export default async function handler(req: IncomingMessage, res: ServerResponse) {
  if (req.method !== "POST") {
    res.statusCode = 405;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: "Method not allowed" }));
    return;
  }
  try {
    const body = await readJsonBody(req);
    const data = await generateExecutiveNarrative(body);
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(data));
  } catch (err: unknown) {
    res.statusCode = 500;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: err instanceof Error ? err.message : "Server error" }));
  }
}
