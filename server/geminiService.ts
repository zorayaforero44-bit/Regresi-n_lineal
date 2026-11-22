import { GoogleGenAI, Type, ThinkingLevel } from "@google/genai";

function getClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY no está configurada en las variables de entorno del servidor.");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

async function withRetry<T>(fn: () => Promise<T>, maxRetries = 2): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt < maxRetries) {
        const delayMs = 600 * Math.pow(2, attempt);
        await new Promise((res) => setTimeout(res, delayMs));
      }
    }
  }
  throw lastError;
}

export interface ContextAnalysisRequest {
  businessContext: {
    sector: string;
    recordUnit: string;
    businessQuestion: string;
    targetOutcome: string;
    horizon: string;
    controllableVars: string;
    constraints: string;
  };
  columns: Array<{
    name: string;
    type: string;
    unit?: string;
    description?: string;
    missingPct: number;
    uniqueCount: number;
    mean?: number;
    min?: number;
    max?: number;
  }>;
  rowCount: number;
}

export async function analyzeBusinessContext(payload: ContextAnalysisRequest) {
  const ai = getClient();

  const prompt = `
Analiza el siguiente problema empresarial y la estructura de variables (sin asumir causalidad automáticamente):

CONTEXTO DEL NEGOCIO:
- Sector: ${payload.businessContext.sector || "No especificado"}
- Unidad de registro (qué representa cada fila): ${payload.businessContext.recordUnit || "No especificado"}
- Pregunta empresarial principal: ${payload.businessContext.businessQuestion || "No especificada"}
- Resultado que desea explicar o pronosticar: ${payload.businessContext.targetOutcome || "No especificado"}
- Horizonte de análisis: ${payload.businessContext.horizon || "No especificado"}
- Variables modificables por el negocio: ${payload.businessContext.controllableVars || "No especificadas"}
- Restricciones operativas/financieras: ${payload.businessContext.constraints || "No especificadas"}
- Total de registros en la base: ${payload.rowCount}

METADATOS DE COLUMNAS:
${JSON.stringify(payload.columns, null, 2)}

INSTRUCCIONES:
1. Sintetiza el problema analítico en lenguaje ejecutivo claro (distinguiendo si el objetivo es explicativo o predictivo).
2. Propón las mejores variables candidatas como Variable Dependiente (Y) considerando tipo de dato, viabilidad para regresión lineal continua (si una candidata es binaria o categórica, advierte explícitamente por qué no aplica regresión lineal clásica y sugiere regresión logística) y coherencia con la pregunta empresarial.
3. Clasifica TODAS las columnas restantes como 'Recomendada', 'Opcional' o 'No recomendada' para actuar como predictores (X), indicando razones de negocio, disponibilidad temporal, riesgo de fuga de información (data leakage) o redundancia.
4. Señala riesgos metodológicos o restricciones del negocio a vigilar durante el modelado y la simulación.
`.trim();

  return withRetry(async () => {
    const response = await ai.models.generateContent({
      model: "gemini-4-hydrogen",
      contents: prompt,
      config: {
        systemInstruction:
          "Eres un equipo experto conformado por un econometrista sénior, científico de datos y economista empresarial. Responde estrictamente en español profesional, con rigor estadístico, sin inventar cifras ni atribuir causalidad por correlación.",
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            problemSynthesis: { type: Type.STRING },
            analyticalFocus: { type: Type.STRING },
            suggestedTargets: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  column: { type: Type.STRING },
                  suitability: { type: Type.STRING },
                  reason: { type: Type.STRING },
                },
                required: ["column", "suitability", "reason"],
              },
            },
            predictorAssessments: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  column: { type: Type.STRING },
                  tier: { type: Type.STRING },
                  controllable: { type: Type.BOOLEAN },
                  reason: { type: Type.STRING },
                },
                required: ["column", "tier", "controllable", "reason"],
              },
            },
            keyConstraintsAndRisks: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
          required: [
            "problemSynthesis",
            "analyticalFocus",
            "suggestedTargets",
            "predictorAssessments",
            "keyConstraintsAndRisks",
          ],
        },
      },
    });

    const text = (response.text ?? "{}").trim();
    return JSON.parse(text);
  });
}

export interface ScenarioRecommendationRequest {
  businessQuestion: string;
  sector: string;
  constraints: string;
  targetName: string;
  targetUnit: string;
  modelSummary: {
    name: string;
    method: string;
    nObs: number;
    r2: number;
    adjR2: number;
    outOfSampleRmse: number;
    outOfSampleMae: number;
    fStat: number | null;
    fPValue: number | null;
    maxVif: number;
    isReliable: boolean;
    reliabilityNote: string;
    coefficients: Array<{
      term: string;
      estimate: number;
      pValue: number | null;
      ciLow: number | null;
      ciHigh: number | null;
      vif: number | null;
    }>;
  };
  scenario: {
    name: string;
    type: string;
    predictedY: number;
    baseY: number;
    diffVsBase: number;
    pctVsBase: number;
    piLow: number;
    piHigh: number;
    outOfRangeWarnings: string[];
    inputs: Array<{
      variable: string;
      baseValue: number | string;
      scenarioValue: number | string;
      generationMethod: string;
      unit?: string;
    }>;
  };
}

export async function generateScenarioRecommendation(payload: ScenarioRecommendationRequest) {
  const ai = getClient();

  const prompt = `
Genera una recomendación ejecutiva rigurosa para el escenario "${payload.scenario.name}" apoyándote EXCLUSIVAMENTE en los resultados estadísticos reales entregados abajo.

PREGUNTA DE NEGOCIO: ${payload.businessQuestion || "Explicar y simular el comportamiento de la variable objetivo"}
SECTOR: ${payload.sector || "Empresarial"}
RESTRICCIONES: ${payload.constraints || "Ninguna especificada"}
VARIABLE OBJETIVO (Y): ${payload.targetName} (${payload.targetUnit || "unidades"})

RESUMEN DEL MODELO SELECCIONADO:
${JSON.stringify(payload.modelSummary, null, 2)}

RESULTADOS DEL ESCENARIO CALCULADO:
${JSON.stringify(payload.scenario, null, 2)}

REGLAS ESTRICTAS:
1. Usa ÚNICAMENTE las cifras reales entregadas en el JSON anterior. No inventes datos ni garantices resultados futuros.
2. Diferencia explícitamente entre predicción condicional del modelo, asociación estadística observada y causalidad real.
3. Si el modelo tiene baja confiabilidad (isReliable=false), alto VIF (>10), p-valores altos, o advertencias de extrapolación fuera de rango (outOfRangeWarnings), destácalo con total transparencia.
4. Sigue exactamente la estructura de 5 secciones solicitada: Hallazgo, Interpretación, Recomendación, Impacto esperado y Riesgos.
`.trim();

  return withRetry(async () => {
    const response = await ai.models.generateContent({
      model: "gemini-4-hydrogen",
      contents: prompt,
      config: {
        systemInstruction:
          "Eres un economista y científico de datos asesorando a dirección ejecutiva. Redacta de forma precisa, accionable y estadísticamente honesta en español.",
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            hallazgo: { type: Type.STRING },
            interpretacion: { type: Type.STRING },
            recomendacion: { type: Type.STRING },
            impactoEsperado: { type: Type.STRING },
            riesgos: { type: Type.STRING },
            distincionCausal: { type: Type.STRING },
            nivelConfianza: { type: Type.STRING },
          },
          required: [
            "hallazgo",
            "interpretacion",
            "recomendacion",
            "impactoEsperado",
            "riesgos",
            "distincionCausal",
            "nivelConfianza",
          ],
        },
      },
    });

    const text = (response.text ?? "{}").trim();
    return JSON.parse(text);
  });
}

export interface ExecutiveNarrativeRequest {
  businessQuestion: string;
  sector: string;
  targetName: string;
  dataQualitySummary: {
    rowsOriginal: number;
    rowsClean: number;
    completenessBefore: number;
    completenessAfter: number;
    trainCount: number;
    valCount: number;
    testCount: number;
  };
  bestModel: {
    name: string;
    method: string;
    adjR2: number;
    outOfSampleRmse: number;
    mae: number;
    fPValue: number | null;
    reliabilityNote: string;
  };
  scenariosSummary: Array<{
    name: string;
    predictedY: number;
    pctVsBase: number;
  }>;
}

export async function generateExecutiveNarrative(payload: ExecutiveNarrativeRequest) {
  const ai = getClient();

  const prompt = `
Redacta un resumen ejecutivo conciso (para informe gerencial) basado estrictamente en estos resultados verificados:
${JSON.stringify(payload, null, 2)}

Devuelve:
- executiveSummary: 2 párrafos claros resumiendo respuesta a la pregunta de negocio, desempeño del mejor modelo y hallazgos de escenarios.
- methodologicalNotes: notas clave sobre prevención de fuga de datos, validación fuera de muestra y supuestos.
- actionPlan: 3 acciones concretas priorizadas para la organización con cautela estadística.
`.trim();

  return withRetry(async () => {
    const response = await ai.models.generateContent({
      model: "gemini-4-hydrogen",
      contents: prompt,
      config: {
        systemInstruction:
          "Eres analista econométrico y estratega de negocios. Basa cada afirmación únicamente en los números proporcionados.",
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            executiveSummary: { type: Type.STRING },
            methodologicalNotes: { type: Type.STRING },
            actionPlan: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
          required: ["executiveSummary", "methodologicalNotes", "actionPlan"],
        },
      },
    });

    const text = (response.text ?? "{}").trim();
    return JSON.parse(text);
  });
}
