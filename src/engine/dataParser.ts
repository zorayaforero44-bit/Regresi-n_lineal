import * as XLSX from "xlsx";
import {
  ColumnProfile,
  DataRow,
  DatasetState,
  QualityIssue,
  VariableType,
  BusinessRole,
} from "../types/analytics";
import {
  excessKurtosis,
  iqrBounds,
  levenshteinDistance,
  mean,
  median,
  quantile,
  skewness,
  std,
} from "./mathStats";

export function isNullish(v: unknown): boolean {
  if (v === null || v === undefined) return true;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    return s === "" || s === "na" || s === "n/a" || s === "null" || s === "nan" || s === "none" || s === "-";
  }
  if (typeof v === "number") return !Number.isFinite(v);
  return false;
}

export function parseDirtyNumber(v: unknown): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v !== "string") return null;
  const trimmed = v.trim();
  if (!trimmed) return null;
  // Clean currency symbols, spaces, %, thousand separators intelligently
  const cleaned = trimmed.replace(/[$€£¥%\s]/g, "");
  // Handle European vs US commas: if both . and , exist, assume last is decimal or standard US
  let normalized = cleaned;
  if (cleaned.includes(",") && cleaned.includes(".")) {
    if (cleaned.lastIndexOf(",") > cleaned.lastIndexOf(".")) {
      normalized = cleaned.replace(/\./g, "").replace(",", ".");
    } else {
      normalized = cleaned.replace(/,/g, "");
    }
  } else if (cleaned.includes(",") && !cleaned.includes(".")) {
    // Check if comma is likely decimal (e.g. 12,5) vs thousand (1,200)
    const parts = cleaned.split(",");
    if (parts.length === 2 && parts[1].length !== 3) {
      normalized = cleaned.replace(",", ".");
    } else {
      normalized = cleaned.replace(/,/g, "");
    }
  }
  const num = Number(normalized);
  return Number.isFinite(num) ? num : null;
}

export function isValidDateValue(v: unknown): boolean {
  if (v instanceof Date) return !Number.isNaN(v.getTime());
  if (typeof v !== "string") return false;
  const s = v.trim();
  if (s.length < 6 || /^\d+$/.test(s)) return false;
  // ISO YYYY-MM-DD, YYYY/MM/DD, DD/MM/YYYY, MM-DD-YYYY, YYYY-MM
  const datePattern =
    /^(\d{4}[-/]\d{1,2}([-/]\d{1,2})?|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})(\s+\d{1,2}:\d{2}(:\d{2})?)?$/;
  if (!datePattern.test(s)) return false;
  const parsed = Date.parse(s);
  return !Number.isNaN(parsed) || /^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(s);
}

export function parseDateToTimestamp(v: unknown): number | null {
  if (v instanceof Date) return v.getTime();
  if (typeof v !== "string") return null;
  const s = v.trim();
  if (!s) return null;
  // Handle DD/MM/YYYY explicitly if >12 in first token
  const slashMatch = s.match(/^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$/);
  if (slashMatch) {
    const a = Number(slashMatch[1]);
    const b = Number(slashMatch[2]);
    let y = Number(slashMatch[3]);
    if (y < 100) y += 2000;
    const month = a > 12 ? b - 1 : a - 1;
    const day = a > 12 ? a : b;
    const d = new Date(y, month, day);
    return Number.isNaN(d.getTime()) ? null : d.getTime();
  }
  const t = Date.parse(s);
  return Number.isNaN(t) ? null : t;
}

function inferUnit(colName: string): string {
  const n = colName.toLowerCase();
  if (n.includes("pct") || n.includes("porcentaje") || n.includes("tasa") || n.includes("descuento")) return "%";
  if (n.includes("precio") || n.includes("costo") || n.includes("inversion") || n.includes("venta") || n.includes("recaudo") || n.includes("ingreso") || n.includes("cartera") || n.includes("gasto")) return "USD / Moneda";
  if (n.includes("hora")) return "Horas";
  if (n.includes("dia") || n.includes("antiguedad")) return "Días / Meses";
  if (n.includes("indice") || n.includes("satisfaccion") || n.includes("score")) return "Puntos";
  if (n.includes("trafico") || n.includes("clientes") || n.includes("unidades")) return "Unidades / Conteo";
  return "Unidades";
}

function inferRole(colName: string, type: VariableType): BusinessRole {
  const n = colName.toLowerCase();
  if (n === "id" || n.endsWith("_id") || n.startsWith("id_") || n === "codigo" || type === "textual") {
    return "identifier_exclude";
  }
  if (
    n.includes("inversion") ||
    n.includes("precio") ||
    n.includes("descuento") ||
    n.includes("gasto") ||
    n.includes("capacitacion") ||
    n.includes("publicidad") ||
    n.includes("mantenimiento") ||
    n.includes("canal") ||
    n.includes("turno")
  ) {
    return "controllable";
  }
  return "external_fixed";
}

function detectFuzzyClusters(categories: Array<{ value: string; count: number }>): Array<{ canonical: string; variants: string[] }> {
  const clusters: Array<{ canonical: string; variants: string[] }> = [];
  const used = new Set<string>();
  const sorted = [...categories].sort((a, b) => b.count - a.count);

  for (let i = 0; i < sorted.length; i++) {
    const base = sorted[i].value;
    if (used.has(base)) continue;
    const variants: string[] = [];
    for (let j = i + 1; j < sorted.length; j++) {
      const cand = sorted[j].value;
      if (used.has(cand)) continue;
      const dist = levenshteinDistance(base, cand);
      const sameLower = base.toLowerCase().trim() === cand.toLowerCase().trim();
      if (sameLower || (base.length >= 4 && dist > 0 && dist <= 2)) {
        variants.push(cand);
        used.add(cand);
      }
    }
    if (variants.length > 0) {
      used.add(base);
      clusters.push({ canonical: base, variants });
    }
  }
  return clusters;
}

export function profileDataset(
  fileName: string,
  fileFormat: string,
  rawRows: DataRow[],
  overrides?: Record<string, Partial<Pick<ColumnProfile, "type" | "unit" | "description" | "businessRole">>>
): DatasetState {
  if (rawRows.length === 0) {
    return {
      fileName,
      fileFormat,
      rawRows: [],
      firstFiveRows: [],
      columns: [],
      duplicateRowCount: 0,
      totalMissingCells: 0,
      completenessPct: 0,
      qualityIssues: [
        {
          id: "empty-file",
          severity: "high",
          category: "missing",
          title: "Archivo sin registros tabulares",
          detail: "No se encontraron filas de datos válidas para analizar.",
          count: 0,
        },
      ],
    };
  }

  const colNames = Array.from(
    rawRows.reduce((set, row) => {
      Object.keys(row).forEach((k) => set.add(k));
      return set;
    }, new Set<string>())
  );

  // Detect duplicate rows
  const seenRows = new Set<string>();
  let duplicateRowCount = 0;
  for (const r of rawRows) {
    const sig = JSON.stringify(colNames.map((c) => r[c] ?? null));
    if (seenRows.has(sig)) duplicateRowCount++;
    else seenRows.add(sig);
  }

  let totalMissingCells = 0;
  const qualityIssues: QualityIssue[] = [];

  if (duplicateRowCount > 0) {
    qualityIssues.push({
      id: "dup-rows",
      severity: "medium",
      category: "duplicates",
      title: "Registros duplicados exactos",
      detail: `Se detectaron ${duplicateRowCount} filas repetidas exactamente iguales que podrían sesgar varianzas y tamaños muestrales.`,
      count: duplicateRowCount,
    });
  }

  const columns: ColumnProfile[] = colNames.map((name) => {
    const values = rawRows.map((r) => (r[name] === undefined ? null : r[name]));
    const nonNull = values.filter((v) => !isNullish(v));
    const missingCount = values.length - nonNull.length;
    totalMissingCells += missingCount;
    const missingPct = (missingCount / values.length) * 100;

    const strSet = new Set(nonNull.map((v) => String(v).trim()));
    const uniqueCount = strSet.size;
    const isConstant = uniqueCount <= 1 && nonNull.length > 0;

    let dirtyNumericCount = 0;
    const numericVals: number[] = [];
    let dateValidCount = 0;

    for (const v of nonNull) {
      if (typeof v === "number" && Number.isFinite(v)) {
        numericVals.push(v);
      } else {
        const num = parseDirtyNumber(v);
        if (num !== null) {
          numericVals.push(num);
          if (typeof v === "string" && /[$€£¥%,]/.test(v)) dirtyNumericCount++;
        } else if (isValidDateValue(v)) {
          dateValidCount++;
        }
      }
    }

    const numericRatio = nonNull.length > 0 ? numericVals.length / nonNull.length : 0;
    const dateRatio = nonNull.length > 0 ? dateValidCount / nonNull.length : 0;

    let detectedType: VariableType = "categorical";
    if (nonNull.length === 0) {
      detectedType = "textual";
    } else if (dateRatio >= 0.75 || (name.toLowerCase().includes("fecha") && dateRatio >= 0.4)) {
      detectedType = "temporal";
    } else if (numericRatio >= 0.8) {
      const uniqueNums = new Set(numericVals);
      const allIntegers = numericVals.every((x) => Math.abs(x - Math.round(x)) < 1e-9);
      if (uniqueNums.size === 2 && (uniqueNums.has(0) && uniqueNums.has(1))) {
        detectedType = "binary";
      } else if (allIntegers && uniqueNums.size <= 10 && values.length >= 20) {
        detectedType = "numeric_discrete";
      } else {
        detectedType = "numeric_continuous";
      }
    } else {
      const lowerVals = Array.from(strSet).map((s) => s.toLowerCase());
      if (
        uniqueCount === 2 ||
        (uniqueCount <= 2 && lowerVals.every((s) => ["si", "sí", "no", "true", "false", "0", "1"].includes(s)))
      ) {
        detectedType = "binary";
      } else if (
        lowerVals.some((s) => ["bajo", "medio", "alto", "baja", "media", "alta"].includes(s)) &&
        uniqueCount <= 6
      ) {
        detectedType = "ordinal";
      } else if (uniqueCount > Math.max(25, values.length * 0.55) && nonNull.some((v) => String(v).length > 24)) {
        detectedType = "textual";
      } else {
        detectedType = "categorical";
      }
    }

    const override = overrides?.[name];
    const finalType = override?.type ?? detectedType;

    // Stats & outliers if numeric
    let stats: ColumnProfile["stats"];
    let outlierCount = 0;
    let negativeCount = 0;
    if (
      (finalType === "numeric_continuous" || finalType === "numeric_discrete" || finalType === "binary") &&
      numericVals.length >= 2
    ) {
      const sorted = [...numericVals].sort((a, b) => a - b);
      const bounds = iqrBounds(numericVals);
      outlierCount = numericVals.filter((x) => x < bounds.low || x > bounds.high).length;
      negativeCount = numericVals.filter((x) => x < 0).length;
      stats = {
        n: numericVals.length,
        mean: mean(numericVals),
        median: median(numericVals),
        std: std(numericVals, 1),
        min: sorted[0],
        q1: quantile(sorted, 0.25),
        q3: quantile(sorted, 0.75),
        max: sorted[sorted.length - 1],
        skewness: skewness(numericVals),
        kurtosis: excessKurtosis(numericVals),
      };
    }

    // Categories & fuzzy clusters
    let categories: ColumnProfile["categories"];
    let fuzzyCategoryClusters: ColumnProfile["fuzzyCategoryClusters"];
    if (finalType === "categorical" || finalType === "ordinal" || finalType === "binary") {
      const freq = new Map<string, number>();
      for (const v of nonNull) {
        const key = String(v).trim();
        freq.set(key, (freq.get(key) ?? 0) + 1);
      }
      categories = Array.from(freq.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .map(([value, count]) => ({ value, count, pct: (count / nonNull.length) * 100 }));
      fuzzyCategoryClusters = detectFuzzyClusters(
        Array.from(freq.entries()).map(([value, count]) => ({ value, count }))
      );
    }

    // Quality issue logging per column
    if (missingCount > 0) {
      qualityIssues.push({
        id: `miss-${name}`,
        severity: missingPct > 20 ? "high" : "medium",
        category: "missing",
        column: name,
        title: `Valores faltantes en ${name}`,
        detail: `${missingCount} registros nulos (${missingPct.toFixed(1)}%). Requiere imputación diferenciada posterior al split.`,
        count: missingCount,
      });
    }
    if (dirtyNumericCount > 0) {
      qualityIssues.push({
        id: `dirty-${name}`,
        severity: "medium",
        category: "dirty_format",
        column: name,
        title: `Formato numérico con símbolos en ${name}`,
        detail: `${dirtyNumericCount} valores contienen símbolos monetarios, porcentajes o separadores mixtos.`,
        count: dirtyNumericCount,
      });
    }
    if (isConstant) {
      qualityIssues.push({
        id: `const-${name}`,
        severity: "high",
        category: "constant",
        column: name,
        title: `Variable constante: ${name}`,
        detail: `Contiene un único valor (${Array.from(strSet)[0] ?? "vacío"}). No aporta varianza explicativa en regresión.`,
        count: values.length,
      });
    }
    if (outlierCount > 0) {
      qualityIssues.push({
        id: `out-${name}`,
        severity: "low",
        category: "outliers",
        column: name,
        title: `Valores atípicos (IQR) en ${name}`,
        detail: `${outlierCount} observaciones fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR]. Evaluar contexto antes de transformar.`,
        count: outlierCount,
      });
    }
    if (fuzzyCategoryClusters && fuzzyCategoryClusters.length > 0) {
      qualityIssues.push({
        id: `fuzzy-${name}`,
        severity: "medium",
        category: "misspelled_category",
        column: name,
        title: `Posibles categorías mal escritas en ${name}`,
        detail: fuzzyCategoryClusters
          .map((c) => `"${c.canonical}" ≈ [${c.variants.join(", ")}]`)
          .join("; "),
        count: fuzzyCategoryClusters.reduce((acc, c) => acc + c.variants.length, 0),
      });
    }
    const lowerName = name.toLowerCase();
    if (
      negativeCount > 0 &&
      (lowerName.includes("precio") ||
        lowerName.includes("venta") ||
        lowerName.includes("inversion") ||
        lowerName.includes("trafico") ||
        lowerName.includes("recaudo") ||
        lowerName.includes("edad"))
    ) {
      qualityIssues.push({
        id: `neg-${name}`,
        severity: "high",
        category: "impossible_value",
        column: name,
        title: `Posibles registros imposibles (negativos) en ${name}`,
        detail: `${negativeCount} valores negativos en una magnitud empresarial que normalmente debe ser ≥ 0.`,
        count: negativeCount,
      });
    }

    return {
      name,
      detectedType,
      type: finalType,
      unit: override?.unit ?? inferUnit(name),
      description: override?.description ?? "",
      businessRole: override?.businessRole ?? inferRole(name, finalType),
      missingCount,
      missingPct,
      uniqueCount,
      sampleValues: nonNull.slice(0, 4),
      dirtyNumericCount,
      negativeCount,
      outlierCount,
      isConstant,
      stats,
      categories,
      fuzzyCategoryClusters,
    };
  });

  const totalCells = rawRows.length * Math.max(1, columns.length);
  const completenessPct = Math.max(0, Math.min(100, ((totalCells - totalMissingCells) / totalCells) * 100));

  return {
    fileName,
    fileFormat,
    rawRows,
    firstFiveRows: rawRows.slice(0, 5),
    columns,
    duplicateRowCount,
    totalMissingCells,
    completenessPct,
    qualityIssues,
  };
}

export async function parseUploadedFile(file: File): Promise<DatasetState> {
  const name = file.name;
  const ext = name.split(".").pop()?.toLowerCase() ?? "";

  if (ext === "parquet") {
    return {
      fileName: name,
      fileFormat: "PARQUET",
      rawRows: [],
      firstFiveRows: [],
      columns: [],
      duplicateRowCount: 0,
      totalMissingCells: 0,
      completenessPct: 0,
      qualityIssues: [],
      unstructuredWarning:
        "El archivo Parquet binario (.parquet) requiere decodificación columnar Apache Arrow en servidor dedicado con pyarrow/duckdb. Para procesarlo de forma inmediata y verificable en el navegador/Vercel sin pérdida de precisión, expórtalo a CSV, XLSX o JSON desde Python/pandas (df.to_csv('datos.csv', index=False)) y cárgalo nuevamente.",
    };
  }

  if (ext === "json") {
    const text = await file.text();
    const parsed = JSON.parse(text);
    let rows: DataRow[] = [];
    if (Array.isArray(parsed)) {
      rows = parsed.map((item) => flattenObject(item));
    } else if (parsed && typeof parsed === "object") {
      // Check if there's an array property like { data: [...] }
      const firstArrKey = Object.keys(parsed).find((k) => Array.isArray(parsed[k]));
      if (firstArrKey) {
        rows = parsed[firstArrKey].map((item: unknown) => flattenObject(item));
      } else {
        rows = [flattenObject(parsed)];
      }
    }
    return profileDataset(name, "JSON", rows);
  }

  if (ext === "xlsx" || ext === "xls") {
    const buffer = await file.arrayBuffer();
    const workbook = XLSX.read(buffer, { type: "array", cellDates: true });
    const firstSheetName = workbook.SheetNames[0];
    const sheet = workbook.Sheets[firstSheetName];
    const rawJson = XLSX.utils.sheet_to_json<DataRow>(sheet, { defval: null, raw: false });
    return profileDataset(name, ext.toUpperCase(), rawJson);
  }

  if (ext === "csv" || ext === "tsv" || ext === "txt") {
    const text = await file.text();
    // Attempt structured parse via SheetJS CSV/TSV parser
    const workbook = XLSX.read(text, {
      type: "string",
      FS: ext === "tsv" ? "\t" : undefined,
      raw: false,
    });
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
    const rows = XLSX.utils.sheet_to_json<DataRow>(firstSheet, { defval: null, raw: false });
    if (rows.length === 0 || Object.keys(rows[0] ?? {}).length <= 1) {
      return {
        fileName: name,
        fileFormat: ext.toUpperCase(),
        rawRows: [],
        firstFiveRows: [],
        columns: [],
        duplicateRowCount: 0,
        totalMissingCells: 0,
        completenessPct: 0,
        qualityIssues: [],
        unstructuredWarning:
          "El archivo cargado parece texto no estructurado o carece de delimitadores tabulares consistentes (CSV/TSV/JSON). Convierte el contenido a columnas estructuradas antes de ejecutar regresión lineal.",
      };
    }
    return profileDataset(name, ext.toUpperCase(), rows);
  }

  return {
    fileName: name,
    fileFormat: ext.toUpperCase() || "DESCONOCIDO",
    rawRows: [],
    firstFiveRows: [],
    columns: [],
    duplicateRowCount: 0,
    totalMissingCells: 0,
    completenessPct: 0,
    qualityIssues: [],
    unstructuredWarning: `Formato "${ext || "desconocido"}" no compatible para parseo tabular directo. Utiliza archivos CSV, XLSX, XLS, JSON o TSV.`,
  };
}

function flattenObject(obj: unknown, prefix = ""): DataRow {
  const out: DataRow = {};
  if (obj === null || typeof obj !== "object") {
    out[prefix || "valor"] = obj as string | number | boolean | null;
    return out;
  }
  for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
    const key = prefix ? `${prefix}_${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v) && !(v instanceof Date)) {
      Object.assign(out, flattenObject(v, key));
    } else if (Array.isArray(v)) {
      out[key] = v.join("; ");
    } else if (v instanceof Date) {
      out[key] = v.toISOString().slice(0, 10);
    } else {
      out[key] = v as string | number | boolean | null;
    }
  }
  return out;
}

export interface SampleDatasetSpec {
  id: string;
  title: string;
  sector: string;
  description: string;
  defaultContext: {
    sector: string;
    recordUnit: string;
    businessQuestion: string;
    targetOutcome: string;
    horizon: string;
    controllableVars: string;
    constraints: string;
  };
  rows: DataRow[];
}

export function getSampleDatasets(): SampleDatasetSpec[] {
  // 1. Retail & Marketing Omnicanal (60 months/regions)
  const retailRows: DataRow[] = [];
  const channels = ["Digital", "Tienda Física", "Omnicanal"];
  for (let i = 0; i < 60; i++) {
    const year = 2022 + Math.floor(i / 12);
    const month = (i % 12) + 1;
    const fecha = `${year}-${String(month).padStart(2, "0")}-01`;
    const isQ4 = month >= 10 || month === 12 ? 1 : 0;
    const canal = i === 14 ? "digtal" : i === 29 ? "Digital " : channels[i % 3];
    const inversionPub = Math.round(18000 + i * 260 + (isQ4 ? 6500 : 0) + ((i * 7919) % 4500));
    const precioUnit = Number((48 - (i % 5) * 1.2 + (i > 40 ? 2.5 : 0)).toFixed(2));
    const descuentoPct = Number((4 + (i % 4) * 3 + (isQ4 ? 4 : 0)).toFixed(1));
    const traficoWeb = Math.round(12000 + inversionPub * 0.85 + (isQ4 ? 8000 : 0) + ((i * 3571) % 3200));
    const satisfaccion = Number((3.8 + (i % 7) * 0.14 - descuentoPct * 0.01).toFixed(2));
    const canalBonus = canal.toLowerCase().includes("omni") ? 28000 : canal.toLowerCase().includes("dig") ? 14000 : 0;
    let ventas =
      65000 +
      2.85 * inversionPub -
      2100 * precioUnit +
      1950 * descuentoPct +
      1.15 * traficoWeb +
      12500 * satisfaccion +
      canalBonus +
      (isQ4 ? 22000 : 0) +
      (((i * 104729) % 14000) - 7000);

    if (i === 52) ventas += 85000; // Realistic seasonal outlier
    retailRows.push({
      Fecha: fecha,
      Canal: canal,
      Inversion_Publicidad: i === 7 || i === 23 ? null : inversionPub,
      Precio_Unitario: precioUnit,
      Descuento_Pct: i === 19 ? null : descuentoPct,
      Trafico_Web: traficoWeb,
      Satisfaccion_Cliente: satisfaccion,
      Temporada_Alta: isQ4,
      Ventas_Mensuales: Math.round(ventas),
    });
  }
  // Add 1 exact duplicate row to demonstrate duplicate cleaning
  retailRows.push({ ...retailRows[3] });

  // 2. Finanzas & Recaudo Empresarial (48 periods)
  const financeRows: DataRow[] = [];
  const segments = ["Corporativo", "PyME", "Consumo"];
  for (let i = 0; i < 48; i++) {
    const year = 2023 + Math.floor(i / 12);
    const month = (i % 12) + 1;
    const mes = `${year}-${String(month).padStart(2, "0")}-01`;
    const segmento = i === 11 ? "pyme" : segments[i % 3];
    const carteraVencida = Math.round(240000 + i * 1800 + ((i * 4999) % 35000));
    const tasaInteres = Number((11.5 + (i % 8) * 0.45).toFixed(2));
    const gastoCobranza = Math.round(12000 + carteraVencida * 0.045 + ((i * 2333) % 3500));
    const clientesActivos = Math.round(920 + i * 8 + ((i * 1973) % 75));
    const segBonus = segmento.toLowerCase() === "corporativo" ? 55000 : segmento.toLowerCase() === "pyme" ? 22000 : 0;
    const recaudo =
      45000 +
      0.42 * carteraVencida -
      3800 * tasaInteres +
      3.9 * gastoCobranza +
      135 * clientesActivos +
      segBonus +
      (((i * 9973) % 18000) - 9000);
    financeRows.push({
      Mes: mes,
      Segmento: segmento,
      Cartera_Vencida: i === 9 ? null : carteraVencida,
      Tasa_Interes_Pct: tasaInteres,
      Gasto_Cobranza: gastoCobranza,
      Clientes_Activos: i === 25 ? null : clientesActivos,
      Recaudo_Mensual: Math.round(recaudo),
    });
  }

  // 3. Productividad & Operaciones (45 plant-weeks)
  const opsRows: DataRow[] = [];
  const plantas = ["Planta Norte", "Planta Centro", "Planta Sur"];
  for (let i = 0; i < 45; i++) {
    const planta = i === 8 ? "Planta norte" : plantas[i % 3];
    const turno = i % 2 === 0 ? "Diurno" : "Nocturno";
    const horasCap = Number((12 + (i % 9) * 3.5).toFixed(1));
    const antiguedadMeses = Math.round(18 + (i % 12) * 3.2);
    const costoMant = Math.round(6500 + (i % 7) * 950 + (i * 120));
    const tasaRotacion = Number((14.5 - horasCap * 0.18 + (turno === "Nocturno" ? 2.4 : 0) + (i % 4) * 0.4).toFixed(2));
    const plantaAdj = planta.toLowerCase().includes("centro") ? 6.5 : planta.toLowerCase().includes("norte") ? 3.2 : 0;
    const prodIndice = Number(
      (
        58 +
        0.92 * horasCap +
        0.31 * antiguedadMeses +
        0.0014 * costoMant -
        1.65 * tasaRotacion +
        plantaAdj +
        (((i * 1733) % 70) / 10 - 3.5)
      ).toFixed(2)
    );
    opsRows.push({
      Planta: planta,
      Turno: turno,
      Horas_Capacitacion: i === 15 ? null : horasCap,
      Antiguedad_Meses: antiguedadMeses,
      Costo_Mantenimiento: costoMant,
      Tasa_Rotacion_Pct: tasaRotacion,
      Productividad_Indice: prodIndice,
    });
  }

  return [
    {
      id: "retail_ventas",
      title: "Retail & Marketing Omnicanal",
      sector: "Comercio Minorista & E-Commerce",
      description: "61 registros mensuales con inversión publicitaria, precio, descuentos, tráfico web, satisfacción y ventas.",
      defaultContext: {
        sector: "Retail y Comercio Omnicanal",
        recordUnit: "Desempeño mensual por canal comercial",
        businessQuestion: "¿Cómo influyen la inversión publicitaria, el precio unitario, los descuentos y el tráfico web sobre las ventas mensuales?",
        targetOutcome: "Ventas_Mensuales (USD)",
        horizon: "Próximos 3 a 6 meses",
        controllableVars: "Inversion_Publicidad, Precio_Unitario, Descuento_Pct, Canal",
        constraints: "Mantener precio unitario competitivo entre $40 y $55; no superar descuentos del 25% ni reducir satisfacción bajo 3.5.",
      },
      rows: retailRows,
    },
    {
      id: "finanzas_recaudo",
      title: "Finanzas & Recaudo de Cartera",
      sector: "Servicios Financieros y Crédito",
      description: "48 registros con cartera vencida, tasa de interés, gasto de cobranza, clientes activos y recaudo mensual.",
      defaultContext: {
        sector: "Finanzas y Gestión de Cartera",
        recordUnit: "Consolidado mensual por segmento de cartera",
        businessQuestion: "¿Qué factores explican el recaudo mensual y qué impacto tiene optimizar el gasto de cobranza?",
        targetOutcome: "Recaudo_Mensual (USD)",
        horizon: "Trimestre fiscal siguiente",
        controllableVars: "Gasto_Cobranza, Segmento",
        constraints: "El gasto de cobranza no debe exceder el 8% de la cartera vencida; la tasa de interés está regulada.",
      },
      rows: financeRows,
    },
    {
      id: "operaciones_productividad",
      title: "Operaciones & Productividad Industrial",
      sector: "Manufactura y Operaciones",
      description: "45 registros por planta y turno con horas de capacitación, antigüedad, mantenimiento, rotación e índice de productividad.",
      defaultContext: {
        sector: "Manufactura y Operaciones",
        recordUnit: "Evaluación operativa por planta y turno",
        businessQuestion: "¿Qué variables explican la productividad operativa y cómo incide la capacitación frente a la rotación?",
        targetOutcome: "Productividad_Indice (0-120 pts)",
        horizon: "Semestre operativo",
        controllableVars: "Horas_Capacitacion, Costo_Mantenimiento, Turno",
        constraints: "Capacidad máxima de capacitación de 45 horas/mes por operario; mantener mantenimiento preventivo mínimo de $6,000.",
      },
      rows: opsRows,
    },
  ];
}
