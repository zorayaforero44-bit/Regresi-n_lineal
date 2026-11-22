import "dotenv/config";
import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import {
  analyzeBusinessContext,
  generateScenarioRecommendation,
  generateExecutiveNarrative,
} from "./server/geminiService.ts";

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: "4mb" }));

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, service: "Linear Regression Intelligence" });
  });

  app.post("/api/gemini/analyze-context", async (req, res) => {
    try {
      const result = await analyzeBusinessContext(req.body);
      res.json(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Error analizando contexto con Gemini";
      console.error("[/api/gemini/analyze-context]", err);
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/gemini/recommend", async (req, res) => {
    try {
      const result = await generateScenarioRecommendation(req.body);
      res.json(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Error generando recomendación con Gemini";
      console.error("[/api/gemini/recommend]", err);
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/gemini/executive-summary", async (req, res) => {
    try {
      const result = await generateExecutiveNarrative(req.body);
      res.json(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Error generando informe ejecutivo con Gemini";
      console.error("[/api/gemini/executive-summary]", err);
      res.status(500).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Linear Regression Intelligence running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
