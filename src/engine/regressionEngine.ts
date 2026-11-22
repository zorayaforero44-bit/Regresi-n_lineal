import {
  CoefficientStat,
  ColumnProfile,
  DataRow,
  ModelComparisonRow,
  ModelMethod,
  ModelSelectionResult,
  ResidualDiagnostics,
  SelectionFocus,
  TrainedModel,
  VariableType,
} from "../types/analytics";
import {
  chi2PValue,
  excessKurtosis,
  fDistPValue,
  invertMatrix,
  matmul,
  mean,
  normalQuantile,
  skewness,
  std,
  studentTPValue,
  studentTQuantile,
  transpose,
} from "./mathStats";
import { computeVIF } from "./edaEngine";

interface PredictorSpec {
  column: string;
  type: VariableType;
  categories?: string[];
  referenceCategory?: string;
  trainMean: number;
  trainStd: number;
  trainMin: number;
  trainMax: number;
}

interface EncodedDesign {
  X: number[][]; // without intercept column
  y: number[];
  terms: Array<{
    term: string;
    sourceColumn: string;
    isDummy: boolean;
    referenceCategory?: string;
  }>;
  specs: PredictorSpec[];
}

export function buildPredictorSpecs(trainRows: DataRow[], predictors: string[], colMap: Map<string, ColumnProfile>): PredictorSpec[] {
  return predictors.map((col) => {
    const profile = colMap.get(col);
    const type: VariableType = profile?.type ?? "numeric_continuous";
    if (type === "categorical" || type === "ordinal" || type === "binary") {
      // Check if already numeric 0/1 binary
      const rawVals = trainRows.map((r) => r[col]).filter((v) => v !== null && v !== undefined);
      const allNumeric = rawVals.length > 0 && rawVals.every((v) => typeof v === "number" && Number.isFinite(v));
      if (allNumeric) {
        const nums = rawVals as number[];
        return {
          column: col,
          type,
          trainMean: mean(nums),
          trainStd: std(nums, 0) || 1,
          trainMin: Math.min(...nums),
          trainMax: Math.max(...nums),
        };
      }
      // Categorical string -> one-hot encoding dropping most frequent category as reference
      const counts = new Map<string, number>();
      for (const v of rawVals) {
        const s = String(v).trim();
        counts.set(s, (counts.get(s) ?? 0) + 1);
      }
      const sortedCats = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
      const referenceCategory = sortedCats[0]?.[0] ?? "Ref";
      const categories = sortedCats.map((x) => x[0]).slice(0, 12); // Cap high-cardinality dummies cleanly
      return {
        column: col,
        type,
        categories,
        referenceCategory,
        trainMean: 0,
        trainStd: 1,
        trainMin: 0,
        trainMax: 1,
      };
    }

    const nums = trainRows
      .map((r) => r[col])
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    const m = nums.length > 0 ? mean(nums) : 0;
    const s = nums.length > 1 ? std(nums, 0) || 1 : 1;
    return {
      column: col,
      type,
      trainMean: m,
      trainStd: s,
      trainMin: nums.length > 0 ? Math.min(...nums) : 0,
      trainMax: nums.length > 0 ? Math.max(...nums) : 0,
    };
  });
}

export function encodeRowsWithSpecs(
  rows: DataRow[],
  targetColumn: string,
  specs: PredictorSpec[]
): EncodedDesign {
  const terms: EncodedDesign["terms"] = [];
  for (const sp of specs) {
    if (sp.categories && sp.referenceCategory) {
      for (const cat of sp.categories) {
        if (cat === sp.referenceCategory) continue;
        terms.push({
          term: `${sp.column}[${cat}]`,
          sourceColumn: sp.column,
          isDummy: true,
          referenceCategory: sp.referenceCategory,
        });
      }
    } else {
      terms.push({
        term: sp.column,
        sourceColumn: sp.column,
        isDummy: false,
      });
    }
  }

  const X: number[][] = [];
  const y: number[] = [];

  for (const row of rows) {
    const yVal = row[targetColumn];
    if (typeof yVal !== "number" || !Number.isFinite(yVal)) continue;
    const xRow: number[] = [];
    let valid = true;
    for (const sp of specs) {
      const val = row[sp.column];
      if (sp.categories && sp.referenceCategory) {
        const strVal = val === null || val === undefined ? sp.referenceCategory : String(val).trim();
        for (const cat of sp.categories) {
          if (cat === sp.referenceCategory) continue;
          xRow.push(strVal === cat ? 1 : 0);
        }
      } else {
        if (typeof val !== "number" || !Number.isFinite(val)) {
          valid = false;
          break;
        }
        xRow.push(val);
      }
    }
    if (valid) {
      X.push(xRow);
      y.push(yVal);
    }
  }

  return { X, y, terms, specs };
}

export function predictFromModel(
  model: TrainedModel,
  inputValues: Record<string, number | string>
): { yHat: number; piLow: number; piHigh: number } {
  const intercept = model.coefficients.find((c) => c.isIntercept)?.estimate ?? 0;
  const xVec: number[] = [1];
  let yHat = intercept;

  for (const sp of model.predictorSpecs) {
    const raw = inputValues[sp.column];
    if (sp.categories && sp.referenceCategory) {
      const strVal = String(raw ?? sp.referenceCategory).trim();
      for (const cat of sp.categories) {
        if (cat === sp.referenceCategory) continue;
        const dummyVal = strVal === cat ? 1 : 0;
        xVec.push(dummyVal);
        const coef = model.coefficients.find((c) => c.term === `${sp.column}[${cat}]`);
        if (coef) yHat += coef.estimate * dummyVal;
      }
    } else {
      const numVal = typeof raw === "number" ? raw : Number(raw ?? sp.trainMean);
      xVec.push(numVal);
      const coef = model.coefficients.find((c) => c.term === sp.column);
      if (coef) yHat += coef.estimate * numVal;
    }
  }

  // Exact OLS Prediction Interval: yHat +/- t_{alpha/2, df} * s * sqrt(1 + x0' (X'X)^-1 x0)
  const df = Math.max(1, model.nTrain - model.coefficients.length);
  const tCrit = model.inferenceValid ? studentTQuantile(0.975, df) : 1.96;
  let leverageTerm = 1 / Math.max(1, model.nTrain);
  if (model.xtxInverse && model.xtxInverse.length === xVec.length) {
    let quad = 0;
    for (let i = 0; i < xVec.length; i++) {
      let rowSum = 0;
      for (let j = 0; j < xVec.length; j++) {
        rowSum += model.xtxInverse[i][j] * xVec[j];
      }
      quad += xVec[i] * rowSum;
    }
    if (Number.isFinite(quad) && quad >= 0) leverageTerm = quad;
  }
  const sePred = model.residualStdError * Math.sqrt(1 + leverageTerm);
  const halfWidth = tCrit * sePred;
  return {
    yHat,
    piLow: yHat - halfWidth,
    piHigh: yHat + halfWidth,
  };
}

function fitRawOLS(XnoIntercept: number[][], y: number[]): { beta: number[]; xtxInv: number[][] } | null {
  const n = XnoIntercept.length;
  const p = XnoIntercept[0]?.length ?? 0;
  if (n <= p + 1) return null;

  const X = XnoIntercept.map((row) => [1, ...row]);
  const Xt = transpose(X);
  const XtX = matmul(Xt, X);
  const inv = invertMatrix(XtX);
  if (!inv) return null;

  const Xty = Xt.map((row) => row.reduce((acc, v, i) => acc + v * y[i], 0));
  const beta = inv.map((row) => row.reduce((acc, v, i) => acc + v * Xty[i], 0));
  return { beta, xtxInv: inv };
}

function computeResidualDiagnostics(
  XnoIntercept: number[][],
  y: number[],
  yHat: number[]
): ResidualDiagnostics {
  const n = y.length;
  const residuals = y.map((yi, i) => yi - yHat[i]);

  // Durbin-Watson
  let numDW = 0;
  let denDW = 0;
  for (let i = 0; i < n; i++) {
    denDW += residuals[i] * residuals[i];
    if (i > 0) {
      const diff = residuals[i] - residuals[i - 1];
      numDW += diff * diff;
    }
  }
  const dw = denDW > 0 ? numDW / denDW : 2;
  const dwNote =
    dw < 1.5
      ? "Autocorrelación positiva potencial en residuos (DW < 1.5)"
      : dw > 2.5
      ? "Autocorrelación negativa potencial en residuos (DW > 2.5)"
      : "Sin evidencia fuerte de autocorrelación (1.5 ≤ DW ≤ 2.5)";

  // Breusch-Pagan test: regress e_i^2 on X, LM = n * R^2_aux ~ Chi2(p)
  const e2 = residuals.map((e) => e * e);
  const bpFit = fitRawOLS(XnoIntercept, e2);
  let bpStat = 0;
  let bpP = 1;
  const p = XnoIntercept[0]?.length ?? 0;
  if (bpFit && p >= 1) {
    const e2Mean = mean(e2);
    let sst = 0;
    let sse = 0;
    for (let i = 0; i < n; i++) {
      let pred = bpFit.beta[0];
      for (let j = 0; j < p; j++) pred += bpFit.beta[j + 1] * XnoIntercept[i][j];
      sse += (e2[i] - pred) ** 2;
      sst += (e2[i] - e2Mean) ** 2;
    }
    const r2Aux = sst > 0 ? Math.max(0, 1 - sse / sst) : 0;
    bpStat = n * r2Aux;
    bpP = chi2PValue(bpStat, p);
  }
  const bpNote =
    bpP < 0.05
      ? "Posible heterocedasticidad (p < 0.05): considerar errores estándar robustos o transformación."
      : "Homocedasticidad compatible con supuestos OLS (p ≥ 0.05).";

  // Jarque-Bera test: JB = (n/6) * (S^2 + K^2 / 4) ~ Chi2(2)
  const sk = skewness(residuals);
  const ku = excessKurtosis(residuals);
  const jbStat = (n / 6) * (sk * sk + (ku * ku) / 4);
  const jbP = chi2PValue(jbStat, 2);
  const jbNote =
    jbP < 0.05
      ? "Residuos se desvían de normalidad (p < 0.05); intervalos de predicción asintóticos requieren cautela en muestras pequeñas."
      : "Normalidad de residuos no rechazada (p ≥ 0.05).";

  // Q-Q Points
  const rStd = std(residuals, 1) || 1;
  const sortedStdRes = [...residuals].map((e) => e / rStd).sort((a, b) => a - b);
  const qqPoints = sortedStdRes.map((val, i) => ({
    theoretical: normalQuantile((i + 0.5) / n),
    sample: val,
  }));

  const residualsVsFitted = yHat.map((fitted, i) => ({
    fitted,
    residual: residuals[i],
    actual: y[i],
  }));

  return {
    durbinWatson: dw,
    durbinWatsonNote: dwNote,
    breuschPaganStat: bpStat,
    breuschPaganPValue: bpP,
    breuschPaganNote: bpNote,
    jarqueBeraStat: jbStat,
    jarqueBeraPValue: jbP,
    jarqueBeraNote: jbNote,
    residualsVsFitted,
    qqPoints,
  };
}

function evalPredictions(X: number[][], y: number[], beta: number[]): { rmse: number; mae: number; r2: number } {
  if (X.length === 0) return { rmse: 0, mae: 0, r2: 0 };
  const yMean = mean(y);
  let sse = 0;
  let sae = 0;
  let sst = 0;
  for (let i = 0; i < X.length; i++) {
    let pred = beta[0];
    for (let j = 0; j < X[i].length; j++) pred += beta[j + 1] * X[i][j];
    const err = y[i] - pred;
    sse += err * err;
    sae += Math.abs(err);
    sst += (y[i] - yMean) ** 2;
  }
  const n = X.length;
  return {
    rmse: Math.sqrt(sse / n),
    mae: sae / n,
    r2: sst > 0 ? 1 - sse / sst : 0,
  };
}

function crossValidateOLS(
  rows: DataRow[],
  targetColumn: string,
  predictors: string[],
  colMap: Map<string, ColumnProfile>,
  isTemporal: boolean
): { cvRmseMean: number; cvMethod: string } {
  const n = rows.length;
  if (n < 12) {
    return { cvRmseMean: 0, cvMethod: "Muestra insuficiente para CV" };
  }
  const kFolds = Math.min(5, Math.max(3, Math.floor(n / 6)));
  const foldRmses: number[] = [];

  if (isTemporal) {
    // Expanding window time-series cross-validation (strictly past -> future)
    const minTrain = Math.max(8, Math.floor(n * 0.45));
    const step = Math.max(1, Math.floor((n - minTrain) / kFolds));
    for (let splitAt = minTrain; splitAt < n - 1; splitAt += step) {
      const trRows = rows.slice(0, splitAt);
      const teRows = rows.slice(splitAt, Math.min(n, splitAt + step));
      const specs = buildPredictorSpecs(trRows, predictors, colMap);
      const trDesign = encodeRowsWithSpecs(trRows, targetColumn, specs);
      const teDesign = encodeRowsWithSpecs(teRows, targetColumn, specs);
      const fit = fitRawOLS(trDesign.X, trDesign.y);
      if (fit && teDesign.X.length > 0) {
        foldRmses.push(evalPredictions(teDesign.X, teDesign.y, fit.beta).rmse);
      }
    }
    return {
      cvRmseMean: foldRmses.length > 0 ? mean(foldRmses) : 0,
      cvMethod: `Validación temporal expansiva (${foldRmses.length} ventanas)`,
    };
  }

  // K-Fold CV
  const foldSize = Math.floor(n / kFolds);
  for (let f = 0; f < kFolds; f++) {
    const start = f * foldSize;
    const end = f === kFolds - 1 ? n : start + foldSize;
    const teRows = rows.slice(start, end);
    const trRows = [...rows.slice(0, start), ...rows.slice(end)];
    const specs = buildPredictorSpecs(trRows, predictors, colMap);
    const trDesign = encodeRowsWithSpecs(trRows, targetColumn, specs);
    const teDesign = encodeRowsWithSpecs(teRows, targetColumn, specs);
    const fit = fitRawOLS(trDesign.X, trDesign.y);
    if (fit && teDesign.X.length > 0) {
      foldRmses.push(evalPredictions(teDesign.X, teDesign.y, fit.beta).rmse);
    }
  }
  return {
    cvRmseMean: foldRmses.length > 0 ? mean(foldRmses) : 0,
    cvMethod: `Validación cruzada ${kFolds}-Fold`,
  };
}

export function trainOLSModel(
  id: string,
  name: string,
  method: "OLS_SIMPLE" | "OLS_MULTIPLE",
  trainRows: DataRow[],
  valRows: DataRow[],
  testRows: DataRow[],
  targetColumn: string,
  predictors: string[],
  colMap: Map<string, ColumnProfile>,
  isTemporal: boolean
): TrainedModel | null {
  const specs = buildPredictorSpecs(trainRows, predictors, colMap);
  const tr = encodeRowsWithSpecs(trainRows, targetColumn, specs);
  const va = encodeRowsWithSpecs(valRows, targetColumn, specs);
  const te = encodeRowsWithSpecs(testRows, targetColumn, specs);

  const fit = fitRawOLS(tr.X, tr.y);
  if (!fit) return null;

  const n = tr.X.length;
  const p = tr.terms.length; // excluding intercept
  const dfModel = p;
  const dfResid = n - p - 1;
  if (dfResid <= 0) return null;

  const yMean = mean(tr.y);
  const yHatTrain: number[] = [];
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    let pred = fit.beta[0];
    for (let j = 0; j < p; j++) pred += fit.beta[j + 1] * tr.X[i][j];
    yHatTrain.push(pred);
    const err = tr.y[i] - pred;
    sse += err * err;
    sst += (tr.y[i] - yMean) ** 2;
  }

  const r2Train = sst > 0 ? Math.max(0, 1 - sse / sst) : 0;
  const adjR2Train = 1 - ((1 - r2Train) * (n - 1)) / dfResid;
  const mseResid = sse / dfResid;
  const residualStdError = Math.sqrt(mseResid);
  const rmseTrain = Math.sqrt(sse / n);

  const fStat = p > 0 && mseResid > 0 ? ((sst - sse) / dfModel) / mseResid : 0;
  const fPValue = fDistPValue(fStat, dfModel, dfResid);

  const aic = n * Math.log(Math.max(1e-12, sse / n)) + 2 * (p + 1);
  const bic = n * Math.log(Math.max(1e-12, sse / n)) + Math.log(n) * (p + 1);

  const valEval = va.X.length > 0 ? evalPredictions(va.X, va.y, fit.beta) : { rmse: rmseTrain, mae: 0, r2: r2Train };
  const testEval = te.X.length > 0 ? evalPredictions(te.X, te.y, fit.beta) : valEval;
  const cv = crossValidateOLS([...trainRows, ...valRows], targetColumn, predictors, colMap, isTemporal);

  // VIF on numeric predictors
  const numericPreds = predictors.filter((c) => {
    const tp = colMap.get(c)?.type;
    return tp === "numeric_continuous" || tp === "numeric_discrete";
  });
  const vifMap = new Map(computeVIF(trainRows, numericPreds).map((v) => [v.column, v.vif]));
  const maxVif = vifMap.size > 0 ? Math.max(...Array.from(vifMap.values())) : 1;

  const tCrit = studentTQuantile(0.975, dfResid);
  const coefficients: CoefficientStat[] = [
    {
      term: "Intercepto (β₀)",
      sourceColumn: "Intercepto",
      isIntercept: true,
      isDummy: false,
      estimate: fit.beta[0],
      stdError: Math.sqrt(Math.max(0, mseResid * fit.xtxInv[0][0])),
      tStat: null,
      pValue: null,
      ciLow: null,
      ciHigh: null,
      vif: null,
      sign: fit.beta[0] > 0 ? "+" : fit.beta[0] < 0 ? "-" : "0",
    },
    ...tr.terms.map((tObj, idx) => {
      const est = fit.beta[idx + 1];
      const se = Math.sqrt(Math.max(1e-18, mseResid * fit.xtxInv[idx + 1][idx + 1]));
      const tStat = se > 0 ? est / se : 0;
      const pValue = studentTPValue(tStat, dfResid);
      return {
        term: tObj.term,
        sourceColumn: tObj.sourceColumn,
        isIntercept: false,
        isDummy: tObj.isDummy,
        referenceCategory: tObj.referenceCategory,
        estimate: est,
        stdError: se,
        tStat,
        pValue,
        ciLow: est - tCrit * se,
        ciHigh: est + tCrit * se,
        vif: tObj.isDummy ? null : vifMap.get(tObj.sourceColumn) ?? 1,
        sign: est > 0 ? ("+" as const) : est < 0 ? ("-" as const) : ("0" as const),
      };
    }),
  ];

  // Fill intercept t & p
  const intCoef = coefficients[0];
  if (intCoef.stdError && intCoef.stdError > 0) {
    intCoef.tStat = intCoef.estimate / intCoef.stdError;
    intCoef.pValue = studentTPValue(intCoef.tStat, dfResid);
    intCoef.ciLow = intCoef.estimate - tCrit * intCoef.stdError;
    intCoef.ciHigh = intCoef.estimate + tCrit * intCoef.stdError;
  }

  const eqParts = [
    `${fit.beta[0].toFixed(3)}`,
    ...tr.terms.map((t, i) => {
      const b = fit.beta[i + 1];
      return `${b >= 0 ? "+" : "-"} ${Math.abs(b).toFixed(4)}·${t.term}`;
    }),
  ];
  const equation = `${targetColumn} = ${eqParts.join(" ")}`;
  const diagnostics = computeResidualDiagnostics(tr.X, tr.y, yHatTrain);

  const reliabilityReasons: string[] = [];
  if (adjR2Train < 0.15) reliabilityReasons.push(`CR² muy bajo (${(adjR2Train * 100).toFixed(1)}%)`);
  if (fPValue > 0.05) reliabilityReasons.push(`Prueba F global no significativa (p=${fPValue.toFixed(4)})`);
  if (maxVif >= 10) reliabilityReasons.push(`Multicolinealidad severa detectada (VIF máx=${maxVif.toFixed(1)})`);
  const isReliable = adjR2Train >= 0.15 && fPValue <= 0.1;

  return {
    id,
    name,
    method,
    targetColumn,
    selectedPredictors: predictors,
    encodedTerms: tr.terms.map((t) => t.term),
    equation,
    nTrain: n,
    nVal: va.X.length,
    nTest: te.X.length,
    r2Train,
    adjR2Train,
    rmseTrain,
    rmseVal: valEval.rmse,
    rmseTest: testEval.rmse,
    maeVal: valEval.mae,
    maeTest: testEval.mae,
    cvRmseMean: cv.cvRmseMean || valEval.rmse,
    cvMethod: cv.cvMethod,
    fStat,
    fPValue,
    aic,
    bic,
    maxVif,
    residualStdError,
    coefficients,
    diagnostics,
    inferenceValid: true,
    signCoherenceNote:
      maxVif >= 10
        ? "Signos de coeficientes pueden invertirse por alta colinealidad (VIF ≥ 10); verificar coherencia económica."
        : "Colinealidad controlada; verificar coherencia económica de signos frente a teoría del negocio.",
    isReliable,
    reliabilityReasons,
    predictorSpecs: specs,
    xtxInverse: fit.xtxInv,
  };
}

export function trainRegularizedModel(
  id: string,
  name: string,
  method: "RIDGE" | "LASSO",
  alpha: number,
  trainRows: DataRow[],
  valRows: DataRow[],
  testRows: DataRow[],
  targetColumn: string,
  predictors: string[],
  colMap: Map<string, ColumnProfile>
): TrainedModel | null {
  const specs = buildPredictorSpecs(trainRows, predictors, colMap);
  const tr = encodeRowsWithSpecs(trainRows, targetColumn, specs);
  const va = encodeRowsWithSpecs(valRows, targetColumn, specs);
  const te = encodeRowsWithSpecs(testRows, targetColumn, specs);

  const n = tr.X.length;
  const p = tr.terms.length;
  if (n <= 4 || p === 0) return null;

  // Standardize X and center y strictly on Train
  const yMean = mean(tr.y);
  const yc = tr.y.map((v) => v - yMean);
  const xMeans = new Array(p).fill(0);
  const xStds = new Array(p).fill(1);
  for (let j = 0; j < p; j++) {
    const colVals = tr.X.map((row) => row[j]);
    xMeans[j] = mean(colVals);
    xStds[j] = std(colVals, 0) || 1;
  }

  const Z: number[][] = tr.X.map((row) => row.map((val, j) => (val - xMeans[j]) / xStds[j]));
  let wStd = new Array(p).fill(0);

  if (method === "RIDGE") {
    const Zt = transpose(Z);
    const ZtZ = matmul(Zt, Z);
    for (let j = 0; j < p; j++) ZtZ[j][j] += alpha * n;
    const inv = invertMatrix(ZtZ);
    if (!inv) return null;
    const Zty = Zt.map((row) => row.reduce((acc, v, i) => acc + v * yc[i], 0));
    wStd = inv.map((row) => row.reduce((acc, v, i) => acc + v * Zty[i], 0));
  } else {
    // Lasso Coordinate Descent
    const zColNormSq = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Z[i][j] * Z[i][j];
      zColNormSq[j] = s || 1;
    }
    const r = [...yc];
    for (let iter = 0; iter < 300; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        const oldW = wStd[j];
        let rho = 0;
        for (let i = 0; i < n; i++) {
          const rNoJ = r[i] + Z[i][j] * oldW;
          rho += Z[i][j] * rNoJ;
        }
        const thresh = n * alpha;
        let newW = 0;
        if (rho > thresh) newW = (rho - thresh) / zColNormSq[j];
        else if (rho < -thresh) newW = (rho + thresh) / zColNormSq[j];
        const delta = newW - oldW;
        if (delta !== 0) {
          for (let i = 0; i < n; i++) r[i] -= Z[i][j] * delta;
          wStd[j] = newW;
          if (Math.abs(delta) > maxDelta) maxDelta = Math.abs(delta);
        }
      }
      if (maxDelta < 1e-6) break;
    }
  }

  // Unscale coefficients back to original feature units
  const betaNoInt = wStd.map((w, j) => w / xStds[j]);
  const intercept = yMean - betaNoInt.reduce((acc, b, j) => acc + b * xMeans[j], 0);
  const beta = [intercept, ...betaNoInt];

  const yHatTrain: number[] = [];
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    let pred = beta[0];
    for (let j = 0; j < p; j++) pred += beta[j + 1] * tr.X[i][j];
    yHatTrain.push(pred);
    const err = tr.y[i] - pred;
    sse += err * err;
    sst += (tr.y[i] - yMean) ** 2;
  }

  const activeParams = betaNoInt.filter((b) => Math.abs(b) > 1e-8).length;
  const dfResid = Math.max(1, n - activeParams - 1);
  const r2Train = sst > 0 ? Math.max(0, 1 - sse / sst) : 0;
  const adjR2Train = 1 - ((1 - r2Train) * (n - 1)) / dfResid;
  const rmseTrain = Math.sqrt(sse / n);
  const residualStdError = Math.sqrt(sse / dfResid);

  const valEval = va.X.length > 0 ? evalPredictions(va.X, va.y, beta) : { rmse: rmseTrain, mae: 0, r2: r2Train };
  const testEval = te.X.length > 0 ? evalPredictions(te.X, te.y, beta) : valEval;

  const coefficients: CoefficientStat[] = [
    {
      term: "Intercepto (β₀)",
      sourceColumn: "Intercepto",
      isIntercept: true,
      isDummy: false,
      estimate: intercept,
      stdError: null,
      tStat: null,
      pValue: null,
      ciLow: null,
      ciHigh: null,
      vif: null,
      sign: intercept > 0 ? "+" : intercept < 0 ? "-" : "0",
    },
    ...tr.terms.map((tObj, idx) => ({
      term: tObj.term,
      sourceColumn: tObj.sourceColumn,
      isIntercept: false,
      isDummy: tObj.isDummy,
      referenceCategory: tObj.referenceCategory,
      estimate: betaNoInt[idx],
      stdError: null,
      tStat: null,
      pValue: null,
      ciLow: null,
      ciHigh: null,
      vif: null,
      sign: betaNoInt[idx] > 1e-8 ? ("+" as const) : betaNoInt[idx] < -1e-8 ? ("-" as const) : ("0" as const),
    })),
  ];

  const eqParts = [
    `${intercept.toFixed(3)}`,
    ...tr.terms
      .map((t, i) => ({ t, b: betaNoInt[i] }))
      .filter((x) => Math.abs(x.b) > 1e-8)
      .map((x) => `${x.b >= 0 ? "+" : "-"} ${Math.abs(x.b).toFixed(4)}·${x.t}`),
  ];

  const diagnostics = computeResidualDiagnostics(tr.X, tr.y, yHatTrain);
  const isReliable = adjR2Train >= 0.15;

  return {
    id,
    name,
    method,
    targetColumn,
    selectedPredictors: predictors,
    encodedTerms: tr.terms.map((t) => t.term),
    equation: `${targetColumn} = ${eqParts.join(" ")}`,
    nTrain: n,
    nVal: va.X.length,
    nTest: te.X.length,
    r2Train,
    adjR2Train,
    rmseTrain,
    rmseVal: valEval.rmse,
    rmseTest: testEval.rmse,
    maeVal: valEval.mae,
    maeTest: testEval.mae,
    cvRmseMean: valEval.rmse,
    cvMethod: "Validación Hold-Out (regularización)",
    fStat: null,
    fPValue: null,
    aic: null,
    bic: null,
    maxVif: 1,
    residualStdError,
    coefficients,
    diagnostics,
    regularizationAlpha: alpha,
    inferenceValid: false,
    inferenceWarning: `Modelo regularizado (${method} α=${alpha}): no se presentan p-valores ni pruebas t clásicas de OLS porque la penalización introduce sesgo de contracción sin inferencia estándar directa.`,
    signCoherenceNote:
      method === "LASSO"
        ? `Lasso contrajo ${p - activeParams} término(s) a cero para prevenir sobreajuste.`
        : "Ridge estabiliza coeficientes ante multicolinealidad mediante penalización L2.",
    isReliable,
    reliabilityReasons: isReliable ? [] : [`CR² bajo (${(adjR2Train * 100).toFixed(1)}%)`],
    predictorSpecs: specs,
  };
}

export function buildCandidateModels(
  trainRows: DataRow[],
  valRows: DataRow[],
  testRows: DataRow[],
  targetColumn: string,
  selectedPredictors: string[],
  colMap: Map<string, ColumnProfile>,
  isTemporal: boolean,
  includeRegularization: boolean
): TrainedModel[] {
  const models: TrainedModel[] = [];
  if (selectedPredictors.length === 0) return models;

  // 1. Multiple OLS with all selected predictors
  const fullOls = trainOLSModel(
    "ols_full",
    `OLS Múltiple Completo (${selectedPredictors.length} var.)`,
    "OLS_MULTIPLE",
    trainRows,
    valRows,
    testRows,
    targetColumn,
    selectedPredictors,
    colMap,
    isTemporal
  );
  if (fullOls) models.push(fullOls);

  // 2. Simple OLS models for top 2 individual predictors (by lowest p-value or best single R²)
  const simpleCandidates: TrainedModel[] = [];
  for (const pred of selectedPredictors.slice(0, 6)) {
    const simple = trainOLSModel(
      `ols_simple_${pred}`,
      `OLS Simple: ${pred}`,
      "OLS_SIMPLE",
      trainRows,
      valRows,
      testRows,
      targetColumn,
      [pred],
      colMap,
      isTemporal
    );
    if (simple) simpleCandidates.push(simple);
  }
  simpleCandidates.sort((a, b) => a.rmseVal - b.rmseVal);
  models.push(...simpleCandidates.slice(0, 2));

  // 3. Parsimonious OLS (keep statistically significant predictors p <= 0.10 from Full OLS, if fewer than full)
  if (fullOls && selectedPredictors.length >= 2) {
    const sigCols = Array.from(
      new Set(
        fullOls.coefficients
          .filter((c) => !c.isIntercept && c.pValue !== null && c.pValue <= 0.1)
          .map((c) => c.sourceColumn)
      )
    );
    if (sigCols.length >= 1 && sigCols.length < selectedPredictors.length) {
      const parsimonious = trainOLSModel(
        "ols_parsimonious",
        `OLS Depurado (Significativas p≤0.10 · ${sigCols.length} var.)`,
        "OLS_MULTIPLE",
        trainRows,
        valRows,
        testRows,
        targetColumn,
        sigCols,
        colMap,
        isTemporal
      );
      if (parsimonious) models.push(parsimonious);
    }
  }

  // 4. Low-VIF OLS if maxVif >= 5
  if (fullOls && fullOls.maxVif >= 5 && selectedPredictors.length >= 3) {
    const highVifCols = new Set(
      fullOls.coefficients.filter((c) => c.vif !== null && c.vif >= 10).map((c) => c.sourceColumn)
    );
    if (highVifCols.size > 0) {
      const dropWorst = Array.from(highVifCols)[0];
      const reducedCols = selectedPredictors.filter((c) => c !== dropWorst);
      if (reducedCols.length >= 1) {
        const lowVifOls = trainOLSModel(
          "ols_low_vif",
          `OLS Sin Colinealidad Severa (sin ${dropWorst})`,
          "OLS_MULTIPLE",
          trainRows,
          valRows,
          testRows,
          targetColumn,
          reducedCols,
          colMap,
          isTemporal
        );
        if (lowVifOls) models.push(lowVifOls);
      }
    }
  }

  // 5. Optional complementary Ridge & Lasso
  if (includeRegularization && selectedPredictors.length >= 2) {
    const yStd = std(
      trainRows.map((r) => r[targetColumn]).filter((v): v is number => typeof v === "number"),
      1
    );
    const ridge = trainRegularizedModel(
      "ridge_l2",
      "Ridge (L2 · Complementario)",
      "RIDGE",
      0.5,
      trainRows,
      valRows,
      testRows,
      targetColumn,
      selectedPredictors,
      colMap
    );
    if (ridge) models.push(ridge);

    const lassoAlpha = Math.max(0.05, Number((yStd * 0.04).toFixed(3)));
    const lasso = trainRegularizedModel(
      "lasso_l1",
      "Lasso (L1 · Complementario)",
      "LASSO",
      lassoAlpha,
      trainRows,
      valRows,
      testRows,
      targetColumn,
      selectedPredictors,
      colMap
    );
    if (lasso) models.push(lasso);
  }

  return models;
}

export function evaluateAndCompareModels(
  models: TrainedModel[],
  focus: SelectionFocus
): ModelSelectionResult {
  if (models.length === 0) {
    return {
      focus,
      recommendedModelId: null,
      noReliableModel: true,
      headline: "No fue posible ajustar modelos con la configuración actual.",
      rationale: "Verifica que existan suficientes registros completos y al menos un predictor seleccionado.",
      advantages: [],
      limitations: ["Sin modelos válidos entrenados."],
      contributingVariables: [],
      attentionAssumptions: [],
      comparisonRows: [],
    };
  }

  const comparisonRows: ModelComparisonRow[] = models.map((m) => {
    const nonInt = m.coefficients.filter((c) => !c.isIntercept);
    const sigCount = nonInt.filter((c) => c.pValue !== null && c.pValue < 0.05).length;
    const sigRatio = m.inferenceValid ? `${sigCount}/${nonInt.length} (p<0.05)` : "N/A (Regularizado)";

    const bpOk = m.diagnostics.breuschPaganPValue >= 0.05;
    const jbOk = m.diagnostics.jarqueBeraPValue >= 0.05;
    const dwOk = m.diagnostics.durbinWatson >= 1.5 && m.diagnostics.durbinWatson <= 2.5;
    const healthCount = [bpOk, jbOk, dwOk].filter(Boolean).length;
    const residualHealth =
      healthCount === 3 ? "Óptimo (3/3 supuestos)" : healthCount === 2 ? "Aceptable (2/3)" : "Requiere atención";

    return {
      modelId: m.id,
      name: m.name,
      method: m.method,
      adjR2: m.adjR2Train,
      rmseVal: m.rmseVal,
      cvRmse: m.cvRmseMean,
      maeVal: m.maeVal,
      significantCoefRatio: sigRatio,
      fPValue: m.fPValue,
      maxVif: m.maxVif,
      residualHealth,
      interpretability: m.method === "OLS_SIMPLE" || m.method === "OLS_MULTIPLE" ? "Alta" : "Media",
      priorityNotes: m.isReliable ? "Desempeño válido" : m.reliabilityReasons.join("; "),
    };
  });

  // Transparent, rule-based multicriteria selection (NOT a fabricated universal score)
  const reliableModels = models.filter((m) => m.isReliable);
  const pool = reliableModels.length > 0 ? reliableModels : models;

  let chosen: TrainedModel;
  if (focus === "predictive") {
    // Prioritize lowest out-of-sample RMSE (Validation / CV), breaking ties by higher CR²
    chosen = [...pool].sort((a, b) => {
      const rmseDiff = a.rmseVal - b.rmseVal;
      if (Math.abs(rmseDiff) > 1e-6) return rmseDiff;
      return b.adjR2Train - a.adjR2Train;
    })[0];
  } else if (focus === "explanatory") {
    // Prioritize OLS models with valid t/p-value inference, low VIF, and highest CR²
    const olsPool = pool.filter((m) => m.inferenceValid);
    const expPool = olsPool.length > 0 ? olsPool : pool;
    chosen = [...expPool].sort((a, b) => {
      const aVifPenalty = a.maxVif >= 10 ? 0.15 : 0;
      const bVifPenalty = b.maxVif >= 10 ? 0.15 : 0;
      const scoreA = a.adjR2Train - aVifPenalty;
      const scoreB = b.adjR2Train - bVifPenalty;
      if (Math.abs(scoreB - scoreA) > 0.015) return scoreB - scoreA;
      return a.rmseVal - b.rmseVal;
    })[0];
  } else {
    // Balanced: Pareto rank across CR², Out-of-sample RMSE, and Valid OLS inference
    const minRmse = Math.min(...pool.map((m) => m.rmseVal)) || 1;
    const maxAdjR2 = Math.max(...pool.map((m) => m.adjR2Train)) || 1;
    chosen = [...pool].sort((a, b) => {
      const gapA = (a.rmseVal - minRmse) / minRmse + (maxAdjR2 - a.adjR2Train);
      const gapB = (b.rmseVal - minRmse) / minRmse + (maxAdjR2 - b.adjR2Train);
      const infBonusA = a.inferenceValid && a.maxVif < 10 ? -0.05 : 0;
      const infBonusB = b.inferenceValid && b.maxVif < 10 ? -0.05 : 0;
      return gapA + infBonusA - (gapB + infBonusB);
    })[0];
  }

  const noReliableModel = reliableModels.length === 0;
  const nonInt = chosen.coefficients.filter((c) => !c.isIntercept);
  const sigVars = nonInt
    .filter((c) => (chosen.inferenceValid ? c.pValue !== null && c.pValue < 0.05 : Math.abs(c.estimate) > 1e-8))
    .map((c) => c.term);

  const advantages: string[] = [];
  if (chosen.adjR2Train >= 0.6) {
    advantages.push(`Alta capacidad explicativa ajustada (CR² = ${(chosen.adjR2Train * 100).toFixed(1)}%).`);
  } else if (chosen.adjR2Train >= 0.3) {
    advantages.push(`Capacidad explicativa moderada (CR² = ${(chosen.adjR2Train * 100).toFixed(1)}%) con control de complejidad.`);
  }
  advantages.push(`Error predictivo fuera de muestra verificado: RMSE Validación = ${chosen.rmseVal.toFixed(2)}, CV = ${chosen.cvRmseMean.toFixed(2)}.`);
  if (chosen.inferenceValid) {
    advantages.push(`Inferencia clásica OLS válida (${sigVars.length} de ${nonInt.length} términos significativos al 5%).`);
  }

  const limitations: string[] = [];
  if (!chosen.inferenceValid && chosen.inferenceWarning) limitations.push(chosen.inferenceWarning);
  if (chosen.maxVif >= 5) limitations.push(`Multicolinealidad ${chosen.maxVif >= 10 ? "severa" : "moderada"} (VIF máx = ${chosen.maxVif.toFixed(2)}).`);
  if (chosen.nTrain < 30) limitations.push(`Tamaño de entrenamiento acotado (n=${chosen.nTrain}); márgenes de error más amplios.`);
  if (limitations.length === 0) limitations.push("Ninguna limitación crítica estructural detectada; mantener monitoreo de supuestos en nuevos datos.");

  const attentionAssumptions: string[] = [];
  if (chosen.diagnostics.breuschPaganPValue < 0.05) attentionAssumptions.push(chosen.diagnostics.breuschPaganNote);
  if (chosen.diagnostics.jarqueBeraPValue < 0.05) attentionAssumptions.push(chosen.diagnostics.jarqueBeraNote);
  if (chosen.diagnostics.durbinWatson < 1.5 || chosen.diagnostics.durbinWatson > 2.5) {
    attentionAssumptions.push(chosen.diagnostics.durbinWatsonNote);
  }
  if (attentionAssumptions.length === 0) {
    attentionAssumptions.push("Homocedasticidad, normalidad e independencia de residuos dentro de rangos aceptables.");
  }

  return {
    focus,
    recommendedModelId: chosen.id,
    noReliableModel,
    headline: noReliableModel
      ? `Advertencia: ningún modelo alcanzó confiabilidad estadística suficiente (mejor CR² = ${(chosen.adjR2Train * 100).toFixed(1)}%)`
      : `Modelo recomendado: ${chosen.name}`,
    rationale: noReliableModel
      ? "Los predictores disponibles no explican suficientemente la variabilidad de la variable objetivo o no superan las pruebas de significancia global. No se recomienda tomar decisiones estratégicas de alto riesgo solo con este ajuste."
      : focus === "predictive"
      ? `Seleccionado por obtener el menor error predictivo fuera de muestra (RMSE Validación = ${chosen.rmseVal.toFixed(2)}) manteniendo coherencia de CR² (${(chosen.adjR2Train * 100).toFixed(1)}%).`
      : focus === "explanatory"
      ? `Seleccionado por ofrecer el mejor equilibrio entre CR² (${(chosen.adjR2Train * 100).toFixed(1)}%), inferencia individual (pruebas t y p-valores válidos) y estabilidad de coeficientes.`
      : `Seleccionado por balancear error fuera de muestra (RMSE = ${chosen.rmseVal.toFixed(2)}), ajuste ajustado (CR² = ${(chosen.adjR2Train * 100).toFixed(1)}%) e interpretabilidad ejecutiva.`,
    advantages,
    limitations,
    contributingVariables: sigVars.length > 0 ? sigVars : nonInt.map((c) => c.term),
    attentionAssumptions,
    comparisonRows,
  };
}
