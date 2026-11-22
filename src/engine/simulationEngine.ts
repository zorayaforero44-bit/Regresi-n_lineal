import {
  ColumnProfile,
  DataRow,
  ScenarioInputItem,
  ScenarioPresetType,
  ScenarioResult,
  SensitivityBar,
  SimulatedRowRecord,
  SimulationStrategyType,
  TrainedModel,
  VariableSimulationPlan,
} from "../types/analytics";
import {
  boxMullerNormal,
  createSeededRng,
  invertMatrix,
  matmul,
  mean,
  median,
  pearsonCorr,
  quantile,
  std,
  transpose,
} from "./mathStats";
import { parseDateToTimestamp } from "./dataParser";
import { predictFromModel } from "./regressionEngine";

const STRATEGY_LABELS: Record<SimulationStrategyType, string> = {
  user_controlled: "Valor controlado por el usuario",
  aux_ols_simple: "Regresión lineal simple auxiliar",
  aux_ols_multiple: "Regresión múltiple auxiliar",
  time_trend: "Pronóstico por tendencia temporal",
  bootstrap: "Bootstrap empírico histórico",
  monte_carlo_normal: "Simulación Monte Carlo paramétrica",
  categorical_empirical: "Muestreo categórico empírico",
};

function fitSimpleLinear(x: number[], y: number[]): { b0: number; b1: number; r2: number; rmse: number } {
  const n = Math.min(x.length, y.length);
  if (n < 3) return { b0: mean(y), b1: 0, r2: 0, rmse: std(y, 1) };
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my);
    den += (x[i] - mx) ** 2;
  }
  const b1 = den > 0 ? num / den : 0;
  const b0 = my - b1 * mx;
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    const err = y[i] - (b0 + b1 * x[i]);
    sse += err * err;
    sst += (y[i] - my) ** 2;
  }
  return {
    b0,
    b1,
    r2: sst > 0 ? Math.max(0, 1 - sse / sst) : 0,
    rmse: Math.sqrt(sse / n),
  };
}

function fitMultiLinearAux(Xraw: number[][], y: number[]): { beta: number[]; r2: number; rmse: number } | null {
  const n = Xraw.length;
  const p = Xraw[0]?.length ?? 0;
  if (n <= p + 2 || p === 0) return null;
  const X = Xraw.map((r) => [1, ...r]);
  const Xt = transpose(X);
  const inv = invertMatrix(matmul(Xt, X));
  if (!inv) return null;
  const Xty = Xt.map((row) => row.reduce((acc, v, i) => acc + v * y[i], 0));
  const beta = inv.map((row) => row.reduce((acc, v, i) => acc + v * Xty[i], 0));
  const my = mean(y);
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    let pred = beta[0];
    for (let j = 0; j < p; j++) pred += beta[j + 1] * Xraw[i][j];
    sse += (y[i] - pred) ** 2;
    sst += (y[i] - my) ** 2;
  }
  return {
    beta,
    r2: sst > 0 ? Math.max(0, 1 - sse / sst) : 0,
    rmse: Math.sqrt(sse / n),
  };
}

export function analyzeVariablesForSimulation(
  cleanRows: DataRow[],
  model: TrainedModel,
  colMap: Map<string, ColumnProfile>,
  dateColumn?: string
): VariableSimulationPlan[] {
  const predictors = model.selectedPredictors;
  const numericPredictors = predictors.filter((c) => {
    const t = colMap.get(c)?.type;
    return t === "numeric_continuous" || t === "numeric_discrete";
  });

  // Sort copy by date if temporal column exists
  const orderedRows = [...cleanRows];
  if (dateColumn) {
    orderedRows.sort((a, b) => (parseDateToTimestamp(a[dateColumn]) ?? 0) - (parseDateToTimestamp(b[dateColumn]) ?? 0));
  }

  return predictors.map((col) => {
    const profile = colMap.get(col);
    const type = profile?.type ?? "numeric_continuous";
    const unit = profile?.unit ?? "Unidades";
    const controllable = profile?.businessRole === "controllable";

    if (type === "categorical" || type === "ordinal" || (type === "binary" && typeof orderedRows[0]?.[col] === "string")) {
      const counts = new Map<string, number>();
      let validN = 0;
      for (const r of orderedRows) {
        const v = r[col];
        if (v !== null && v !== undefined) {
          const s = String(v).trim();
          counts.set(s, (counts.get(s) ?? 0) + 1);
          validN++;
        }
      }
      const categories = Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([value, c]) => ({ value, prob: validN > 0 ? c / validN : 0 }));

      const recommendedStrategy: SimulationStrategyType = controllable ? "user_controlled" : "categorical_empirical";
      return {
        column: col,
        type,
        unit,
        controllable,
        seasonalityDetected: false,
        correlatedWith: [],
        categories,
        recommendedStrategy,
        selectedStrategy: recommendedStrategy,
        justification: controllable
          ? "Variable categórica/discreta modificable por decisión comercial; no corresponde imponer regresión lineal sobre categorías."
          : "Variable categórica discreta: se modela mediante distribución empírica histórica de frecuencias.",
        auxFitQuality: `${categories.length} categorías observadas`,
        uncertaintySummary: `Moda: ${categories[0]?.value ?? "—"} (${((categories[0]?.prob ?? 0) * 100).toFixed(1)}%)`,
        availableStrategies: [
          { value: "user_controlled", label: STRATEGY_LABELS.user_controlled },
          { value: "categorical_empirical", label: STRATEGY_LABELS.categorical_empirical },
        ],
        sparkline: categories.map((c) => Math.round(c.prob * 100)),
      };
    }

    const vals = orderedRows
      .map((r) => r[col])
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    const hMean = vals.length > 0 ? mean(vals) : 0;
    const hMedian = vals.length > 0 ? median(vals) : 0;
    const hStd = vals.length > 1 ? std(vals, 1) : 0;
    const hMin = vals.length > 0 ? Math.min(...vals) : 0;
    const hMax = vals.length > 0 ? Math.max(...vals) : 0;

    // Trend analysis over index/time
    const tIdx = vals.map((_, i) => i + 1);
    const trendFit = fitSimpleLinear(tIdx, vals);
    const seasonalityDetected = Boolean(
      dateColumn && vals.length >= 12 && hStd > 0 && Math.abs(pearsonCorr(vals.slice(0, -1), vals.slice(1))) > 0.35
    );

    // Correlations with other numeric predictors
    const correlatedWith: Array<{ column: string; r: number }> = [];
    for (const other of numericPredictors) {
      if (other === col) continue;
      const pairedX: number[] = [];
      const pairedY: number[] = [];
      for (const r of orderedRows) {
        const vx = r[other];
        const vy = r[col];
        if (typeof vx === "number" && typeof vy === "number") {
          pairedX.push(vx);
          pairedY.push(vy);
        }
      }
      if (pairedX.length >= 5) {
        correlatedWith.push({ column: other, r: pearsonCorr(pairedX, pairedY) });
      }
    }
    correlatedWith.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));

    // Auxiliary regression check
    let bestAuxSimpleR2 = 0;
    let bestDriver = "";
    if (correlatedWith.length > 0) {
      bestDriver = correlatedWith[0].column;
      bestAuxSimpleR2 = correlatedWith[0].r ** 2;
    }

    let auxMultiR2 = 0;
    const multiDrivers = correlatedWith.slice(0, 3).map((c) => c.column);
    if (multiDrivers.length >= 2) {
      const Xaux: number[][] = [];
      const yaux: number[] = [];
      for (const r of orderedRows) {
        const yv = r[col];
        if (typeof yv === "number" && multiDrivers.every((d) => typeof r[d] === "number")) {
          Xaux.push(multiDrivers.map((d) => r[d] as number));
          yaux.push(yv);
        }
      }
      const mFit = fitMultiLinearAux(Xaux, yaux);
      if (mFit) auxMultiR2 = mFit.r2;
    }

    // Choose recommended strategy without forcing regression on all variables
    let recommendedStrategy: SimulationStrategyType = "bootstrap";
    let justification = "";
    let auxFitQuality = "";
    let auxR2: number | undefined;
    let auxDrivers: string[] | undefined;

    if (controllable) {
      recommendedStrategy = "user_controlled";
      justification =
        "Palanca controlable por el negocio según diccionario: se recomienda definirla directamente como decisión gerencial en escenarios.";
      auxFitQuality = `Rango histórico [${hMin.toFixed(1)}, ${hMax.toFixed(1)}]`;
    } else if (auxMultiR2 >= 0.45 && multiDrivers.length >= 2 && auxMultiR2 > bestAuxSimpleR2 + 0.08) {
      recommendedStrategy = "aux_ols_multiple";
      auxR2 = auxMultiR2;
      auxDrivers = multiDrivers;
      justification = `Presenta dependencia conjunta con ${multiDrivers.join(", ")} (R² auxiliar = ${(auxMultiR2 * 100).toFixed(1)}%), preservando coherencia estructural.`;
      auxFitQuality = `R² auxiliar múltiple = ${(auxMultiR2 * 100).toFixed(1)}%`;
    } else if (bestAuxSimpleR2 >= 0.35 && bestDriver) {
      recommendedStrategy = "aux_ols_simple";
      auxR2 = bestAuxSimpleR2;
      auxDrivers = [bestDriver];
      justification = `Fuerte relación lineal con ${bestDriver} (r = ${correlatedWith[0].r.toFixed(2)}, R² aux = ${(bestAuxSimpleR2 * 100).toFixed(1)}%); se proyecta condicionada a su impulsor.`;
      auxFitQuality = `R² auxiliar simple = ${(bestAuxSimpleR2 * 100).toFixed(1)}%`;
    } else if (dateColumn && trendFit.r2 >= 0.3) {
      recommendedStrategy = "time_trend";
      auxR2 = trendFit.r2;
      justification = `Tendencia temporal consistente detectada (R² temporal = ${(trendFit.r2 * 100).toFixed(1)}%, pendiente ${trendFit.b1 >= 0 ? "+" : ""}${trendFit.b1.toFixed(2)}/periodo).`;
      auxFitQuality = `R² tendencia = ${(trendFit.r2 * 100).toFixed(1)}%`;
    } else if (Math.abs(profile?.stats?.skewness ?? 0) > 0.7 || type === "numeric_discrete" || type === "binary") {
      recommendedStrategy = "bootstrap";
      justification = "Distribución asimétrica, acotada o discreta: el bootstrap empírico respeta fielmente la forma histórica sin asumir normalidad.";
      auxFitQuality = `Resampling empírico (n=${vals.length})`;
    } else {
      recommendedStrategy = "monte_carlo_normal";
      justification = "Comportamiento estable aproximadamente simétrico: simulación Monte Carlo acotada a límites históricos plausibles.";
      auxFitQuality = `σ = ${hStd.toFixed(2)} (CV ${hMean !== 0 ? ((hStd / Math.abs(hMean)) * 100).toFixed(1) : "0"}%)`;
    }

    const availableStrategies: Array<{ value: SimulationStrategyType; label: string }> = [
      { value: "user_controlled", label: STRATEGY_LABELS.user_controlled },
      { value: "bootstrap", label: STRATEGY_LABELS.bootstrap },
      { value: "monte_carlo_normal", label: STRATEGY_LABELS.monte_carlo_normal },
    ];
    if (bestDriver) availableStrategies.push({ value: "aux_ols_simple", label: STRATEGY_LABELS.aux_ols_simple });
    if (multiDrivers.length >= 2) availableStrategies.push({ value: "aux_ols_multiple", label: STRATEGY_LABELS.aux_ols_multiple });
    if (dateColumn) availableStrategies.push({ value: "time_trend", label: STRATEGY_LABELS.time_trend });

    const step = Math.max(1, Math.floor(vals.length / 18));
    const sparkline = vals.filter((_, i) => i % step === 0).slice(0, 18);

    return {
      column: col,
      type,
      unit,
      controllable,
      historicalMean: hMean,
      historicalMedian: hMedian,
      historicalStd: hStd,
      historicalMin: hMin,
      historicalMax: hMax,
      trendSlope: trendFit.b1,
      trendR2: trendFit.r2,
      seasonalityDetected,
      correlatedWith: correlatedWith.slice(0, 3),
      recommendedStrategy,
      selectedStrategy: recommendedStrategy,
      justification,
      auxFitQuality,
      auxR2,
      auxDrivers,
      viableMin: hMin,
      viableMax: hMax,
      uncertaintySummary: `Media ${hMean.toFixed(2)} ± ${hStd.toFixed(2)} | Rango [${hMin.toFixed(1)}, ${hMax.toFixed(1)}]`,
      availableStrategies,
      sparkline,
    };
  });
}

function clampToViable(val: number, min?: number, max?: number, isDiscrete?: boolean): number {
  let out = val;
  if (min !== undefined && max !== undefined) {
    const pad = (max - min) * 0.15;
    out = Math.max(min - pad, Math.min(max + pad, out));
  }
  if (isDiscrete) out = Math.round(out);
  return Number(out.toFixed(2));
}

export function generateSimulatedDataset(
  cleanRows: DataRow[],
  plans: VariableSimulationPlan[],
  model: TrainedModel,
  horizonCount = 12,
  seed = 2026
): SimulatedRowRecord[] {
  const rng = createSeededRng(seed);
  const records: SimulatedRowRecord[] = [];
  const planMap = new Map(plans.map((p) => [p.column, p]));

  // Pre-fit auxiliary regressions on cleanRows
  const auxSimpleModels = new Map<string, { driver: string; b0: number; b1: number; rmse: number }>();
  const auxMultiModels = new Map<string, { drivers: string[]; beta: number[]; rmse: number }>();
  const trendModels = new Map<string, { b0: number; b1: number; rmse: number; nHist: number }>();

  for (const plan of plans) {
    const vals = cleanRows
      .map((r) => r[plan.column])
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (vals.length >= 3) {
      const tIdx = vals.map((_, i) => i + 1);
      const tf = fitSimpleLinear(tIdx, vals);
      trendModels.set(plan.column, { b0: tf.b0, b1: tf.b1, rmse: tf.rmse, nHist: vals.length });
    }
    if (plan.auxDrivers && plan.auxDrivers.length >= 1) {
      const drv = plan.auxDrivers[0];
      const xs: number[] = [];
      const ys: number[] = [];
      for (const r of cleanRows) {
        if (typeof r[drv] === "number" && typeof r[plan.column] === "number") {
          xs.push(r[drv] as number);
          ys.push(r[plan.column] as number);
        }
      }
      const sf = fitSimpleLinear(xs, ys);
      auxSimpleModels.set(plan.column, { driver: drv, b0: sf.b0, b1: sf.b1, rmse: sf.rmse });
    }
    if (plan.auxDrivers && plan.auxDrivers.length >= 2) {
      const drvs = plan.auxDrivers;
      const Xaux: number[][] = [];
      const yaux: number[] = [];
      for (const r of cleanRows) {
        if (typeof r[plan.column] === "number" && drvs.every((d) => typeof r[d] === "number")) {
          Xaux.push(drvs.map((d) => r[d] as number));
          yaux.push(r[plan.column] as number);
        }
      }
      const mf = fitMultiLinearAux(Xaux, yaux);
      if (mf) auxMultiModels.set(plan.column, { drivers: drvs, beta: mf.beta, rmse: mf.rmse });
    }
  }

  // Order simulation: non-auxiliary drivers first, then auxiliary-dependent variables
  const independentPlans = plans.filter(
    (p) => p.selectedStrategy !== "aux_ols_simple" && p.selectedStrategy !== "aux_ols_multiple"
  );
  const dependentPlans = plans.filter(
    (p) => p.selectedStrategy === "aux_ols_simple" || p.selectedStrategy === "aux_ols_multiple"
  );

  for (let h = 1; h <= horizonCount; h++) {
    const values: Record<string, number | string> = {};

    for (const plan of independentPlans) {
      const strat = plan.selectedStrategy;
      if (plan.categories && plan.categories.length > 0) {
        if (strat === "user_controlled") {
          values[plan.column] = plan.categories[0].value;
        } else {
          const u = rng();
          let cum = 0;
          let picked = plan.categories[0].value;
          for (const c of plan.categories) {
            cum += c.prob;
            if (u <= cum) {
              picked = c.value;
              break;
            }
          }
          values[plan.column] = picked;
        }
        continue;
      }

      const histVals = cleanRows
        .map((r) => r[plan.column])
        .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
      const isDisc = plan.type === "numeric_discrete" || plan.type === "binary";

      if (strat === "user_controlled") {
        // Gentle planned variation around historical median
        const base = plan.historicalMedian ?? plan.historicalMean ?? 0;
        values[plan.column] = isDisc ? Math.round(base) : Number(base.toFixed(2));
      } else if (strat === "time_trend") {
        const tm = trendModels.get(plan.column);
        const raw = tm ? tm.b0 + tm.b1 * (tm.nHist + h) + boxMullerNormal(rng) * tm.rmse * 0.5 : plan.historicalMean ?? 0;
        values[plan.column] = clampToViable(raw, plan.viableMin, plan.viableMax, isDisc);
      } else if (strat === "bootstrap" && histVals.length > 0) {
        const pick = histVals[Math.floor(rng() * histVals.length)];
        values[plan.column] = isDisc ? Math.round(pick) : Number(pick.toFixed(2));
      } else {
        const m = plan.historicalMean ?? 0;
        const s = plan.historicalStd ?? 1;
        const raw = m + boxMullerNormal(rng) * s;
        values[plan.column] = clampToViable(raw, plan.viableMin, plan.viableMax, isDisc);
      }
    }

    for (const plan of dependentPlans) {
      const isDisc = plan.type === "numeric_discrete" || plan.type === "binary";
      if (plan.selectedStrategy === "aux_ols_multiple") {
        const mf = auxMultiModels.get(plan.column);
        if (mf) {
          let pred = mf.beta[0];
          mf.drivers.forEach((d, idx) => {
            const dv = values[d] ?? planMap.get(d)?.historicalMean ?? 0;
            pred += mf.beta[idx + 1] * Number(dv);
          });
          pred += boxMullerNormal(rng) * mf.rmse * 0.6;
          values[plan.column] = clampToViable(pred, plan.viableMin, plan.viableMax, isDisc);
          continue;
        }
      }
      if (plan.selectedStrategy === "aux_ols_simple") {
        const sf = auxSimpleModels.get(plan.column);
        if (sf) {
          const dv = Number(values[sf.driver] ?? planMap.get(sf.driver)?.historicalMean ?? 0);
          const pred = sf.b0 + sf.b1 * dv + boxMullerNormal(rng) * sf.rmse * 0.6;
          values[plan.column] = clampToViable(pred, plan.viableMin, plan.viableMax, isDisc);
          continue;
        }
      }
      const m = plan.historicalMean ?? 0;
      values[plan.column] = isDisc ? Math.round(m) : Number(m.toFixed(2));
    }

    const pred = predictFromModel(model, values);
    records.push({
      periodIndex: h,
      sourceType: "Simulado",
      values,
      predictedY: pred.yHat,
      piLow: pred.piLow,
      piHigh: pred.piHigh,
    });
  }

  return records;
}

export function computeScenarioMatrix(
  cleanRows: DataRow[],
  plans: VariableSimulationPlan[],
  model: TrainedModel,
  customOverrides: Record<string, number | string>,
  mcIterations = 1000,
  seed = 777
): ScenarioResult[] {
  const planMap = new Map(plans.map((p) => [p.column, p]));

  // Base input values: median for numeric, mode for categorical
  const baseValues: Record<string, number | string> = {};
  for (const plan of plans) {
    if (plan.categories && plan.categories.length > 0) {
      baseValues[plan.column] = plan.categories[0].value;
    } else {
      const m = plan.historicalMedian ?? plan.historicalMean ?? 0;
      baseValues[plan.column] = plan.type === "numeric_discrete" || plan.type === "binary" ? Math.round(m) : Number(m.toFixed(2));
    }
  }

  const basePred = predictFromModel(model, baseValues);
  const baseY = basePred.yHat;

  // Determine direction of beneficial effect per predictor from model's estimated coefficients
  const getCoefDirection = (col: string): number => {
    const coef = model.coefficients.find((c) => !c.isIntercept && c.sourceColumn === col && !c.isDummy);
    if (!coef || Math.abs(coef.estimate) < 1e-12) return 1;
    return coef.estimate > 0 ? 1 : -1;
  };

  // Helper to propagate auxiliary regressions when controllable/upstream variables shift
  const propagateCoupled = (draft: Record<string, number | string>) => {
    const out = { ...draft };
    for (const plan of plans) {
      if (plan.selectedStrategy === "aux_ols_simple" && plan.auxDrivers?.[0]) {
        const drv = plan.auxDrivers[0];
        if (customOverrides[plan.column] === undefined || out[plan.column] === baseValues[plan.column]) {
          const xs: number[] = [];
          const ys: number[] = [];
          for (const r of cleanRows) {
            if (typeof r[drv] === "number" && typeof r[plan.column] === "number") {
              xs.push(r[drv] as number);
              ys.push(r[plan.column] as number);
            }
          }
          const sf = fitSimpleLinear(xs, ys);
          const v = sf.b0 + sf.b1 * Number(out[drv]);
          const isDisc = plan.type === "numeric_discrete" || plan.type === "binary";
          out[plan.column] = clampToViable(v, plan.historicalMin, plan.historicalMax, isDisc);
        }
      }
    }
    return out;
  };

  const buildScenario = (
    id: string,
    name: string,
    type: ScenarioPresetType,
    rawValues: Record<string, number | string>
  ): ScenarioResult => {
    const pred = predictFromModel(model, rawValues);
    const outOfRangeWarnings: string[] = [];

    const inputs: ScenarioInputItem[] = plans.map((plan) => {
      const val = rawValues[plan.column] ?? baseValues[plan.column];
      let outOfRange = false;
      if (typeof val === "number" && plan.historicalMin !== undefined && plan.historicalMax !== undefined) {
        if (val < plan.historicalMin || val > plan.historicalMax) {
          outOfRange = true;
          outOfRangeWarnings.push(
            `${plan.column} (${val.toFixed(2)}) excede el rango histórico observado [${plan.historicalMin.toFixed(2)}, ${plan.historicalMax.toFixed(2)}]: riesgo de extrapolación.`
          );
        }
      }
      return {
        column: plan.column,
        type: plan.type,
        unit: plan.unit,
        baseValue: baseValues[plan.column],
        scenarioValue: val,
        generationMethod: plan.selectedStrategy,
        historicalMin: plan.historicalMin,
        historicalMax: plan.historicalMax,
        outOfRange,
      };
    });

    // Sensitivity tornado around this scenario (±0.5 SD or ±10%)
    const sensitivity: SensitivityBar[] = plans
      .filter((p) => !p.categories || p.categories.length === 0)
      .map((p) => {
        const curr = Number(rawValues[p.column] ?? baseValues[p.column]);
        const delta = p.historicalStd && p.historicalStd > 0 ? p.historicalStd * 0.5 : Math.abs(curr) * 0.1 || 1;
        const hiVals = { ...rawValues, [p.column]: curr + delta };
        const loVals = { ...rawValues, [p.column]: curr - delta };
        const yHi = predictFromModel(model, hiVals).yHat - pred.yHat;
        const yLo = predictFromModel(model, loVals).yHat - pred.yHat;
        return {
          column: p.column,
          impactPositive: yHi,
          impactNegative: yLo,
          absSpread: Math.abs(yHi - yLo),
        };
      })
      .sort((a, b) => b.absSpread - a.absSpread)
      .slice(0, 8);

    const diffVsBase = pred.yHat - baseY;
    const pctVsBase = baseY !== 0 ? (diffVsBase / Math.abs(baseY)) * 100 : 0;

    return {
      id,
      name,
      type,
      inputs,
      predictedY: pred.yHat,
      baseY,
      diffVsBase,
      pctVsBase,
      piLow: pred.piLow,
      piHigh: pred.piHigh,
      uncertaintyHalfWidth: (pred.piHigh - pred.piLow) / 2,
      outOfRangeWarnings,
      sensitivity,
    };
  };

  // Shift helper respecting historical bounds and coefficient sign (never assumes +X is always favorable!)
  const makeShiftedValues = (sdFactor: number, favorable: boolean) => {
    const draft: Record<string, number | string> = { ...baseValues };
    for (const plan of plans) {
      if (plan.categories && plan.categories.length > 0) {
        if (plan.controllable && plan.categories.length > 1) {
          // Evaluate which category yields highest/lowest Y
          const catScores = plan.categories.map((c) => ({
            val: c.value,
            y: predictFromModel(model, { ...baseValues, [plan.column]: c.value }).yHat,
          }));
          catScores.sort((a, b) => (favorable ? b.y - a.y : a.y - b.y));
          if (sdFactor >= 0.5) draft[plan.column] = catScores[0].val;
        }
        continue;
      }
      if (plan.selectedStrategy === "aux_ols_simple" || plan.selectedStrategy === "aux_ols_multiple") continue;

      const dir = getCoefDirection(plan.column) * (favorable ? 1 : -1);
      const baseNum = Number(baseValues[plan.column]);
      const step = (plan.historicalStd ?? 0) * sdFactor * dir;
      const isDisc = plan.type === "numeric_discrete" || plan.type === "binary";
      const minB = plan.historicalMin ?? baseNum - Math.abs(step);
      const maxB = plan.historicalMax ?? baseNum + Math.abs(step);
      let candidate = Math.max(minB, Math.min(maxB, baseNum + step));
      if (isDisc) candidate = Math.round(candidate);
      draft[plan.column] = Number(candidate.toFixed(2));
    }
    return propagateCoupled(draft);
  };

  const baseScenario = buildScenario("base", "Escenario Base", "base", baseValues);
  const optScenario = buildScenario("optimista", "Escenario Optimista", "optimista", makeShiftedValues(0.85, true));
  const consScenario = buildScenario("conservador", "Escenario Conservador", "conservador", makeShiftedValues(0.35, true));
  const advScenario = buildScenario("adverso", "Escenario Adverso", "adverso", makeShiftedValues(0.8, false));

  // Custom scenario merges baseValues + customOverrides
  const customMerged = { ...baseValues, ...customOverrides };
  const customScenario = buildScenario("personalizado", "Escenario Personalizado", "personalizado", customMerged);

  // Probabilistic Monte Carlo Scenario
  const simRecords = generateSimulatedDataset(cleanRows, plans, model, mcIterations, seed);
  const rng = createSeededRng(seed + 99);
  const mcOutcomes = simRecords.map((r) => r.predictedY + boxMullerNormal(rng) * model.residualStdError).sort((a, b) => a - b);
  const mcMean = mean(mcOutcomes);
  const mcStd = std(mcOutcomes, 1);
  const p10 = quantile(mcOutcomes, 0.1);
  const p50 = quantile(mcOutcomes, 0.5);
  const p90 = quantile(mcOutcomes, 0.9);
  const mcMin = mcOutcomes[0] ?? mcMean;
  const mcMax = mcOutcomes[mcOutcomes.length - 1] ?? mcMean;

  // Build 12 histogram bins for Monte Carlo outcomes
  const binCount = 12;
  const binW = mcMax > mcMin ? (mcMax - mcMin) / binCount : 1;
  const binCounts = new Array(binCount).fill(0);
  for (const yVal of mcOutcomes) {
    const idx = mcMax > mcMin ? Math.min(binCount - 1, Math.floor((yVal - mcMin) / binW)) : 0;
    binCounts[idx]++;
  }
  const mcHist = binCounts.map((count, i) => ({
    bin: `${(mcMin + i * binW).toFixed(0)}`,
    count,
  }));

  // Mean simulated input values for summary table
  const mcAvgInputs: Record<string, number | string> = { ...baseValues };
  for (const plan of plans) {
    if (!plan.categories || plan.categories.length === 0) {
      mcAvgInputs[plan.column] = Number(mean(simRecords.map((r) => Number(r.values[plan.column] ?? 0))).toFixed(2));
    }
  }

  const mcScenario = buildScenario("probabilistico", `Escenario Probabilístico (${mcIterations} sim.)`, "probabilistico", mcAvgInputs);
  mcScenario.predictedY = mcMean;
  mcScenario.piLow = p10;
  mcScenario.piHigh = p90;
  mcScenario.uncertaintyHalfWidth = (p90 - p10) / 2;
  mcScenario.diffVsBase = mcMean - baseY;
  mcScenario.pctVsBase = baseY !== 0 ? ((mcMean - baseY) / Math.abs(baseY)) * 100 : 0;
  mcScenario.monteCarloSummary = {
    iterations: mcIterations,
    mean: mcMean,
    std: mcStd,
    p10,
    p50,
    p90,
    min: mcMin,
    max: mcMax,
    histogram: mcHist,
  };

  return [baseScenario, optScenario, consScenario, advScenario, customScenario, mcScenario];
}
