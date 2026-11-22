import {
  CleanedDatasetResult,
  CleaningChangeItem,
  CleaningColumnRule,
  ColumnProfile,
  DataRow,
  DatasetState,
  SplitConfig,
} from "../types/analytics";
import { createSeededRng, iqrBounds, mean, median } from "./mathStats";
import { isNullish, isValidDateValue, parseDateToTimestamp, parseDirtyNumber } from "./dataParser";

export function buildRecommendedCleaningRules(
  columns: ColumnProfile[],
  targetColumn: string | null
): CleaningColumnRule[] {
  const catCols = columns.filter(
    (c) => (c.type === "categorical" || c.type === "ordinal" || c.type === "binary") && c.name !== targetColumn
  );
  const numCols = columns.filter(
    (c) => (c.type === "numeric_continuous" || c.type === "numeric_discrete") && c.name !== targetColumn
  );

  return columns.map((col) => {
    const isTarget = col.name === targetColumn;
    let recommendedStrategy: CleaningColumnRule["recommendedStrategy"] = "none";
    let reason = "Sin valores faltantes detectados.";
    let groupByColumn: string | undefined;

    if (isTarget) {
      recommendedStrategy = "drop_rows";
      reason =
        col.missingCount > 0
          ? "Variable objetivo (Y): nunca se imputa con valores predichos para reentrenar; se excluyen registros sin etiqueta válida."
          : "Variable objetivo (Y) completa; cualquier registro nulo futuro será excluido antes del entrenamiento.";
    } else if (col.type === "temporal") {
      recommendedStrategy = col.missingCount > 0 ? "forward_fill" : "none";
      reason =
        col.missingCount > 0
          ? "Variable temporal: arrastre secuencial usando únicamente información pasada (sin fuga futura)."
          : "Secuencia temporal completa.";
    } else if (col.type === "numeric_continuous" || col.type === "numeric_discrete") {
      if (col.missingCount > 0) {
        const skew = Math.abs(col.stats?.skewness ?? 0);
        if (col.missingPct > 4 && numCols.length >= 3) {
          recommendedStrategy = "knn";
          reason = `Faltantes (${col.missingPct.toFixed(1)}%) con múltiples predictores numéricos: KNN multivariado ajustado solo en entrenamiento.`;
        } else if (skew > 0.8 || col.outlierCount > 0) {
          recommendedStrategy = "median";
          reason = `Asimetría (${skew.toFixed(2)}) o ${col.outlierCount} atípicos detectados: la mediana evita distorsión por extremos.`;
        } else {
          recommendedStrategy = "mean";
          reason = "Distribución aproximadamente simétrica sin atípicos severos: media aritmética recomendada.";
        }
      }
    } else if (col.type === "categorical" || col.type === "ordinal" || col.type === "binary") {
      if (col.missingCount > 0) {
        if (col.missingPct >= 10) {
          recommendedStrategy = "missing_category";
          reason = `Ausencia elevada (${col.missingPct.toFixed(1)}%): preservar como categoría explícita "Sin Dato" evita sesgar frecuencias.`;
        } else if (catCols.length > 1) {
          const other = catCols.find((c) => c.name !== col.name && c.missingCount === 0);
          if (other) {
            recommendedStrategy = "group_mode";
            groupByColumn = other.name;
            reason = `Imputación por moda condicionada al grupo "${other.name}" aprendida en entrenamiento.`;
          } else {
            recommendedStrategy = "mode";
            reason = "Imputación por moda del conjunto de entrenamiento.";
          }
        } else {
          recommendedStrategy = "mode";
          reason = "Imputación por moda (categoría más frecuente en entrenamiento).";
        }
      }
    }

    return {
      column: col.name,
      recommendedStrategy,
      selectedStrategy: recommendedStrategy,
      reason,
      groupByColumn,
      standardizeCategories: (col.fuzzyCategoryClusters?.length ?? 0) > 0,
      parseDirtyNumbers: col.dirtyNumericCount > 0,
      outlierAction: "keep_flagged",
    };
  });
}

function getMode(values: string[]): string {
  const freq = new Map<string, number>();
  for (const v of values) freq.set(v, (freq.get(v) ?? 0) + 1);
  let best = "";
  let bestC = -1;
  for (const [k, c] of freq) {
    if (c > bestC) {
      bestC = c;
      best = k;
    }
  }
  return best || "Sin Dato";
}

export function executeCleaningPipeline(
  dataset: DatasetState,
  rules: CleaningColumnRule[],
  splitConfig: SplitConfig,
  targetColumn: string | null,
  deduplicate: boolean
): CleanedDatasetResult {
  const changes: CleaningChangeItem[] = [];
  const ruleMap = new Map(rules.map((r) => [r.column, r]));
  const colMap = new Map(dataset.columns.map((c) => [c.name, c]));

  // Clone rows
  let workingRows: DataRow[] = dataset.rawRows.map((r) => ({ ...r }));

  // 1. Dirty numeric parsing & category standardization (deterministic row-level syntax cleanup, zero cross-row leakage)
  for (const col of dataset.columns) {
    const rule = ruleMap.get(col.name);
    if (!rule) continue;

    if (
      (col.type === "numeric_continuous" || col.type === "numeric_discrete" || col.type === "binary") &&
      (rule.parseDirtyNumbers || col.dirtyNumericCount > 0)
    ) {
      let parsedCount = 0;
      for (const row of workingRows) {
        const val = row[col.name];
        if (!isNullish(val) && typeof val === "string") {
          const num = parseDirtyNumber(val);
          if (num !== null) {
            row[col.name] = num;
            parsedCount++;
          } else {
            row[col.name] = null;
          }
        }
      }
      if (parsedCount > 0) {
        changes.push({
          column: col.name,
          action: "Normalización numérica",
          affectedCount: parsedCount,
          detail: "Conversión de texto con símbolos/comas a valores numéricos puros.",
        });
      }
    } else if (col.type === "numeric_continuous" || col.type === "numeric_discrete") {
      // Ensure numeric strings are cast to numbers
      for (const row of workingRows) {
        const val = row[col.name];
        if (!isNullish(val) && typeof val !== "number") {
          const num = parseDirtyNumber(val);
          row[col.name] = num;
        }
      }
    }

    if (
      (col.type === "categorical" || col.type === "ordinal" || col.type === "binary") &&
      rule.standardizeCategories
    ) {
      const variantToCanonical = new Map<string, string>();
      for (const cluster of col.fuzzyCategoryClusters ?? []) {
        variantToCanonical.set(cluster.canonical.trim(), cluster.canonical.trim());
        for (const v of cluster.variants) {
          variantToCanonical.set(v.trim(), cluster.canonical.trim());
        }
      }
      let fixed = 0;
      for (const row of workingRows) {
        const val = row[col.name];
        if (!isNullish(val)) {
          const rawStr = String(val);
          const trimmed = rawStr.trim();
          const mapped = variantToCanonical.get(trimmed) ?? trimmed;
          if (mapped !== rawStr) {
            row[col.name] = mapped;
            fixed++;
          }
        }
      }
      if (fixed > 0) {
        changes.push({
          column: col.name,
          action: "Estandarización de categorías",
          affectedCount: fixed,
          detail: "Unificación de variantes ortográficas y espacios en blanco.",
        });
      }
    }
  }

  // 2. Deduplication
  let deduplicatedRows = 0;
  if (deduplicate) {
    const seen = new Set<string>();
    const deduped: DataRow[] = [];
    for (const r of workingRows) {
      const sig = JSON.stringify(dataset.columns.map((c) => r[c.name] ?? null));
      if (seen.has(sig)) {
        deduplicatedRows++;
      } else {
        seen.add(sig);
        deduped.push(r);
      }
    }
    workingRows = deduped;
    if (deduplicatedRows > 0) {
      changes.push({
        column: "Dataset completo",
        action: "Eliminación de duplicados",
        affectedCount: deduplicatedRows,
        detail: `${deduplicatedRows} filas duplicadas exactas removidas.`,
      });
    }
  }

  // 3. Target Y missing exclusion (never impute Y to train on imputed Y)
  let droppedTargetRows = 0;
  if (targetColumn) {
    const validRows: DataRow[] = [];
    for (const r of workingRows) {
      if (isNullish(r[targetColumn])) droppedTargetRows++;
      else validRows.push(r);
    }
    workingRows = validRows;
    if (droppedTargetRows > 0) {
      changes.push({
        column: targetColumn,
        action: "Exclusión de Y sin etiqueta válida",
        affectedCount: droppedTargetRows,
        detail: "Se excluyeron registros con Y nulo para evitar fuga o entrenamiento artificial.",
      });
    }
  }

  // 4. Split BEFORE fitting any statistical imputer or scaler
  let orderedRows = [...workingRows];
  if (splitConfig.mode === "temporal" && splitConfig.dateColumn) {
    const dc = splitConfig.dateColumn;
    orderedRows.sort((a, b) => {
      const ta = parseDateToTimestamp(a[dc]) ?? 0;
      const tb = parseDateToTimestamp(b[dc]) ?? 0;
      return ta - tb;
    });
  } else {
    const rng = createSeededRng(splitConfig.seed);
    for (let i = orderedRows.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = orderedRows[i];
      orderedRows[i] = orderedRows[j];
      orderedRows[j] = tmp;
    }
  }

  const totalN = orderedRows.length;
  const nTrain = Math.max(1, Math.round((totalN * splitConfig.trainPct) / 100));
  const nVal = Math.max(0, Math.round((totalN * splitConfig.valPct) / 100));
  const trainRows = orderedRows.slice(0, nTrain).map((r) => ({ ...r }));
  const valRows = orderedRows.slice(nTrain, nTrain + nVal).map((r) => ({ ...r }));
  const testRows = orderedRows.slice(nTrain + nVal).map((r) => ({ ...r }));

  // 5. Fit imputers & outlier caps STRICTLY on trainRows, then apply to train, val, test
  const numericRefCols = dataset.columns
    .filter((c) => (c.type === "numeric_continuous" || c.type === "numeric_discrete") && c.name !== targetColumn)
    .map((c) => c.name);

  // Precompute train column means/stds for KNN Euclidean distance
  const trainMeans = new Map<string, number>();
  const trainStds = new Map<string, number>();
  for (const colName of numericRefCols) {
    const vals = trainRows.map((r) => r[colName]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    const m = vals.length > 0 ? mean(vals) : 0;
    const s = vals.length > 1 ? Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length) || 1 : 1;
    trainMeans.set(colName, m);
    trainStds.set(colName, s);
  }

  for (const col of dataset.columns) {
    const rule = ruleMap.get(col.name);
    if (!rule || col.name === targetColumn) continue;

    let totalImputed = 0;

    if (col.type === "numeric_continuous" || col.type === "numeric_discrete") {
      const trainVals = trainRows
        .map((r) => r[col.name])
        .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
      const trainMean = trainVals.length > 0 ? mean(trainVals) : 0;
      const trainMedian = trainVals.length > 0 ? median(trainVals) : 0;

      // Group medians fit on Train
      const groupMedians = new Map<string, number>();
      if (rule.selectedStrategy === "group_median" && rule.groupByColumn) {
        const grpMap = new Map<string, number[]>();
        for (const r of trainRows) {
          const g = String(r[rule.groupByColumn] ?? "Sin Dato");
          const v = r[col.name];
          if (typeof v === "number" && Number.isFinite(v)) {
            const arr = grpMap.get(g) ?? [];
            arr.push(v);
            grpMap.set(g, arr);
          }
        }
        for (const [g, arr] of grpMap) groupMedians.set(g, median(arr));
      }

      const applyNumericImputation = (rows: DataRow[]) => {
        for (const r of rows) {
          if (!isNullish(r[col.name])) continue;
          if (rule.selectedStrategy === "none") continue;
          if (rule.selectedStrategy === "mean") {
            r[col.name] = Number(trainMean.toFixed(4));
            totalImputed++;
          } else if (rule.selectedStrategy === "median") {
            r[col.name] = Number(trainMedian.toFixed(4));
            totalImputed++;
          } else if (rule.selectedStrategy === "group_median" && rule.groupByColumn) {
            const g = String(r[rule.groupByColumn] ?? "Sin Dato");
            r[col.name] = Number((groupMedians.get(g) ?? trainMedian).toFixed(4));
            totalImputed++;
          } else if (rule.selectedStrategy === "knn") {
            // Find k=5 nearest neighbors in trainRows where target col is non-null
            const candidates: Array<{ dist: number; val: number }> = [];
            for (const tr of trainRows) {
              const tv = tr[col.name];
              if (typeof tv !== "number" || !Number.isFinite(tv)) continue;
              let distSq = 0;
              let usedDims = 0;
              for (const ref of numericRefCols) {
                if (ref === col.name) continue;
                const a = r[ref];
                const b = tr[ref];
                if (typeof a === "number" && typeof b === "number") {
                  const s = trainStds.get(ref) ?? 1;
                  const diff = (a - b) / s;
                  distSq += diff * diff;
                  usedDims++;
                }
              }
              const dist = usedDims > 0 ? Math.sqrt(distSq / usedDims) : 0;
              candidates.push({ dist, val: tv });
            }
            candidates.sort((a, b) => a.dist - b.dist);
            const topK = candidates.slice(0, Math.min(5, candidates.length));
            const knnVal = topK.length > 0 ? mean(topK.map((c) => c.val)) : trainMedian;
            r[col.name] = Number(knnVal.toFixed(4));
            totalImputed++;
          }
        }
      };

      applyNumericImputation(trainRows);
      applyNumericImputation(valRows);
      applyNumericImputation(testRows);

      // Optional IQR winsorization learned strictly on trainRows
      if (rule.outlierAction === "winsorize_iqr" && trainVals.length >= 4) {
        const bounds = iqrBounds(trainVals);
        let winsorized = 0;
        for (const splitArr of [trainRows, valRows, testRows]) {
          for (const r of splitArr) {
            const v = r[col.name];
            if (typeof v === "number") {
              if (v < bounds.low) {
                r[col.name] = Number(bounds.low.toFixed(4));
                winsorized++;
              } else if (v > bounds.high) {
                r[col.name] = Number(bounds.high.toFixed(4));
                winsorized++;
              }
            }
          }
        }
        if (winsorized > 0) {
          changes.push({
            column: col.name,
            action: "Winsorización IQR (umbrales de Train)",
            affectedCount: winsorized,
            detail: `Valores extremos acotados a [${bounds.low.toFixed(2)}, ${bounds.high.toFixed(2)}].`,
          });
        }
      }
    } else if (col.type === "categorical" || col.type === "ordinal" || col.type === "binary") {
      const trainCats = trainRows
        .map((r) => r[col.name])
        .filter((v) => !isNullish(v))
        .map((v) => String(v).trim());
      const trainMode = getMode(trainCats);

      const groupModes = new Map<string, string>();
      if (rule.selectedStrategy === "group_mode" && rule.groupByColumn) {
        const grpMap = new Map<string, string[]>();
        for (const r of trainRows) {
          const g = String(r[rule.groupByColumn] ?? "Sin Dato");
          const v = r[col.name];
          if (!isNullish(v)) {
            const arr = grpMap.get(g) ?? [];
            arr.push(String(v).trim());
            grpMap.set(g, arr);
          }
        }
        for (const [g, arr] of grpMap) groupModes.set(g, getMode(arr));
      }

      const applyCatImputation = (rows: DataRow[]) => {
        for (const r of rows) {
          if (!isNullish(r[col.name])) continue;
          if (rule.selectedStrategy === "none") continue;
          if (rule.selectedStrategy === "missing_category") {
            r[col.name] = "Sin Dato";
            totalImputed++;
          } else if (rule.selectedStrategy === "group_mode" && rule.groupByColumn) {
            const g = String(r[rule.groupByColumn] ?? "Sin Dato");
            r[col.name] = groupModes.get(g) ?? trainMode;
            totalImputed++;
          } else if (rule.selectedStrategy === "mode") {
            r[col.name] = trainMode;
            totalImputed++;
          }
        }
      };

      applyCatImputation(trainRows);
      applyCatImputation(valRows);
      applyCatImputation(testRows);
    } else if (col.type === "temporal") {
      // Past-only forward fill within each split separately
      const applyForwardFill = (rows: DataRow[], fallback: string | null) => {
        let lastValid = fallback;
        for (const r of rows) {
          const v = r[col.name];
          if (!isNullish(v) && isValidDateValue(v)) {
            lastValid = String(v);
          } else if (rule.selectedStrategy === "forward_fill" && lastValid !== null) {
            r[col.name] = lastValid;
            totalImputed++;
          }
        }
        return lastValid;
      };
      const lastTrainDate = applyForwardFill(trainRows, null);
      const lastValDate = applyForwardFill(valRows, lastTrainDate);
      applyForwardFill(testRows, lastValDate);
    }

    if (totalImputed > 0) {
      changes.push({
        column: col.name,
        action: `Imputación (${rule.selectedStrategy})`,
        affectedCount: totalImputed,
        detail: `Parámetros aprendidos exclusivamente sobre Train (n=${trainRows.length}) y aplicados sin fuga a Val/Test.`,
      });
    }
  }

  const allCleanRows = [...trainRows, ...valRows, ...testRows];
  let missingAfter = 0;
  for (const r of allCleanRows) {
    for (const col of dataset.columns) {
      if (isNullish(r[col.name])) missingAfter++;
    }
  }
  const totalCellsAfter = allCleanRows.length * Math.max(1, dataset.columns.length);
  const completenessAfter =
    totalCellsAfter > 0 ? Math.max(0, Math.min(100, ((totalCellsAfter - missingAfter) / totalCellsAfter) * 100)) : 0;

  return {
    allCleanRows,
    trainRows,
    valRows,
    testRows,
    droppedTargetRows,
    deduplicatedRows,
    completenessBefore: dataset.completenessPct,
    completenessAfter,
    changes,
    learnedParametersNote: `Split ${splitConfig.mode === "temporal" ? "cronológico" : "aleatorio"} (${splitConfig.trainPct}% Train / ${splitConfig.valPct}% Val / ${splitConfig.testPct}% Test) ejecutado ANTES de ajustar medias, medianas, modas, KNN y cotas IQR.`,
    splitConfig,
  };
}
