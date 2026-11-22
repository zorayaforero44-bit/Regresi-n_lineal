import {
  CategoricalAssociation,
  ColumnProfile,
  CorrelationMatrixResult,
  CorrelationMethod,
  DataRow,
  VifEntry,
} from "../types/analytics";
import {
  cramersV,
  etaSquared,
  invertMatrix,
  kendallTauB,
  matmul,
  mean,
  pearsonCorr,
  spearmanCorr,
  transpose,
} from "./mathStats";
import { parseDateToTimestamp } from "./dataParser";

export function getNumericValuesPair(rows: DataRow[], colA: string, colB: string): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  for (const r of rows) {
    const va = r[colA];
    const vb = r[colB];
    if (typeof va === "number" && Number.isFinite(va) && typeof vb === "number" && Number.isFinite(vb)) {
      x.push(va);
      y.push(vb);
    }
  }
  return { x, y };
}

export function computeCorrelationMatrix(
  rows: DataRow[],
  numericCols: string[],
  method: CorrelationMethod
): CorrelationMatrixResult {
  const k = numericCols.length;
  const matrix: number[][] = Array.from({ length: k }, () => new Array(k).fill(1));
  const fn = method === "pearson" ? pearsonCorr : method === "spearman" ? spearmanCorr : kendallTauB;

  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const { x, y } = getNumericValuesPair(rows, numericCols[i], numericCols[j]);
      const val = x.length >= 3 ? fn(x, y) : 0;
      matrix[i][j] = val;
      matrix[j][i] = val;
    }
  }
  return { method, columns: numericCols, matrix };
}

export function computeCategoricalAssociations(
  rows: DataRow[],
  columns: ColumnProfile[],
  targetCol: string | null
): CategoricalAssociation[] {
  const catCols = columns.filter(
    (c) => (c.type === "categorical" || c.type === "ordinal" || c.type === "binary") && !c.isConstant
  );
  const numCols = columns.filter(
    (c) => (c.type === "numeric_continuous" || c.type === "numeric_discrete") && !c.isConstant
  );
  const results: CategoricalAssociation[] = [];

  // Categorical vs Categorical -> Cramér's V
  for (let i = 0; i < catCols.length; i++) {
    for (let j = i + 1; j < catCols.length; j++) {
      const a: string[] = [];
      const b: string[] = [];
      for (const r of rows) {
        const va = r[catCols[i].name];
        const vb = r[catCols[j].name];
        if (va !== null && va !== undefined && vb !== null && vb !== undefined) {
          a.push(String(va));
          b.push(String(vb));
        }
      }
      const v = cramersV(a, b);
      results.push({
        colA: catCols[i].name,
        colB: catCols[j].name,
        metric: "Cramér's V",
        value: v,
        interpretation:
          v >= 0.5
            ? "Asociación categórica fuerte (posible redundancia)"
            : v >= 0.25
            ? "Asociación categórica moderada"
            : "Asociación categórica débil",
      });
    }
  }

  // Categorical vs Target Numeric (or top numeric) -> Eta-squared
  const yCol = targetCol && numCols.some((c) => c.name === targetCol) ? targetCol : numCols[0]?.name;
  if (yCol) {
    for (const c of catCols) {
      const cats: string[] = [];
      const nums: number[] = [];
      for (const r of rows) {
        const vc = r[c.name];
        const vy = r[yCol];
        if (vc !== null && vc !== undefined && typeof vy === "number" && Number.isFinite(vy)) {
          cats.push(String(vc));
          nums.push(vy);
        }
      }
      const eta2 = etaSquared(cats, nums);
      results.push({
        colA: c.name,
        colB: yCol,
        metric: "Eta-cuadrado (η²)",
        value: eta2,
        interpretation:
          eta2 >= 0.14
            ? `Explica ${(eta2 * 100).toFixed(1)}% de la varianza univariada de ${yCol} (efecto grande)`
            : eta2 >= 0.06
            ? `Explica ${(eta2 * 100).toFixed(1)}% de la varianza de ${yCol} (efecto medio)`
            : `Efecto univariado bajo (${(eta2 * 100).toFixed(1)}%) sobre ${yCol}`,
      });
    }
  }

  return results;
}

export function computeVIF(rows: DataRow[], predictorCols: string[]): VifEntry[] {
  if (predictorCols.length < 2) {
    return predictorCols.map((c) => ({ column: c, vif: 1, tolerance: 1, severity: "normal" }));
  }

  // Complete cases across predictorCols
  const validRows = rows.filter((r) =>
    predictorCols.every((c) => typeof r[c] === "number" && Number.isFinite(r[c] as number))
  );
  if (validRows.length <= predictorCols.length + 2) {
    return predictorCols.map((c) => ({ column: c, vif: 1, tolerance: 1, severity: "normal" }));
  }

  const results: VifEntry[] = [];
  for (let targetIdx = 0; targetIdx < predictorCols.length; targetIdx++) {
    const yName = predictorCols[targetIdx];
    const otherCols = predictorCols.filter((_, idx) => idx !== targetIdx);
    const y = validRows.map((r) => r[yName] as number);
    const X = validRows.map((r) => [1, ...otherCols.map((c) => r[c] as number)]);

    const Xt = transpose(X);
    const XtX = matmul(Xt, X);
    const inv = invertMatrix(XtX);
    if (!inv) {
      results.push({ column: yName, vif: 999, tolerance: 0.001, severity: "severe" });
      continue;
    }
    const Xty = Xt.map((row) => row.reduce((sum, val, i) => sum + val * y[i], 0));
    const beta = inv.map((row) => row.reduce((sum, val, i) => sum + val * Xty[i], 0));

    const yMean = mean(y);
    let sst = 0;
    let sse = 0;
    for (let i = 0; i < validRows.length; i++) {
      let yHat = 0;
      for (let j = 0; j < X[i].length; j++) yHat += X[i][j] * beta[j];
      const err = y[i] - yHat;
      sse += err * err;
      sst += (y[i] - yMean) * (y[i] - yMean);
    }
    const r2 = sst > 0 ? Math.max(0, Math.min(0.9999, 1 - sse / sst)) : 0;
    const vif = 1 / Math.max(1e-4, 1 - r2);
    results.push({
      column: yName,
      vif,
      tolerance: 1 / vif,
      severity: vif >= 10 ? "severe" : vif >= 5 ? "moderate" : "normal",
    });
  }
  return results;
}

export function buildHistogramBins(values: number[], binCount = 12): Array<{ label: string; count: number; mid: number }> {
  if (values.length === 0) return [];
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  if (minVal === maxVal) {
    return [{ label: minVal.toFixed(1), count: values.length, mid: minVal }];
  }
  const width = (maxVal - minVal) / binCount;
  const counts = new Array(binCount).fill(0);
  for (const v of values) {
    const idx = Math.min(binCount - 1, Math.floor((v - minVal) / width));
    counts[idx]++;
  }
  return counts.map((count, i) => {
    const lo = minVal + i * width;
    const hi = lo + width;
    return {
      label: `${lo.toFixed(0)}–${hi.toFixed(0)}`,
      count,
      mid: (lo + hi) / 2,
    };
  });
}

export function buildTimeSeriesPoints(
  rows: DataRow[],
  dateCol: string,
  metricCol: string
): Array<{ date: string; ts: number; value: number }> {
  const pts: Array<{ date: string; ts: number; value: number }> = [];
  for (const r of rows) {
    const dv = r[dateCol];
    const mv = r[metricCol];
    const ts = parseDateToTimestamp(dv);
    if (ts !== null && typeof mv === "number" && Number.isFinite(mv)) {
      pts.push({ date: String(dv), ts, value: mv });
    }
  }
  pts.sort((a, b) => a.ts - b.ts);
  return pts;
}
