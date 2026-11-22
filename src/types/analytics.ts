export type VariableType =
  | "numeric_continuous"
  | "numeric_discrete"
  | "categorical"
  | "ordinal"
  | "binary"
  | "temporal"
  | "textual";

export type BusinessRole = "controllable" | "external_fixed" | "identifier_exclude";

export type DataRow = Record<string, string | number | boolean | null>;

export interface ColumnProfile {
  name: string;
  detectedType: VariableType;
  type: VariableType;
  unit: string;
  description: string;
  businessRole: BusinessRole;
  missingCount: number;
  missingPct: number;
  uniqueCount: number;
  sampleValues: Array<string | number | boolean | null>;
  dirtyNumericCount: number;
  negativeCount: number;
  outlierCount: number;
  isConstant: boolean;
  stats?: {
    n: number;
    mean: number;
    median: number;
    std: number;
    min: number;
    q1: number;
    q3: number;
    max: number;
    skewness: number;
    kurtosis: number;
  };
  categories?: Array<{ value: string; count: number; pct: number }>;
  fuzzyCategoryClusters?: Array<{ canonical: string; variants: string[] }>;
}

export interface QualityIssue {
  id: string;
  severity: "high" | "medium" | "low";
  category:
    | "missing"
    | "duplicates"
    | "dirty_format"
    | "misspelled_category"
    | "outliers"
    | "constant"
    | "impossible_value"
    | "inconsistent_date";
  column?: string;
  title: string;
  detail: string;
  count: number;
}

export interface DatasetState {
  fileName: string;
  fileFormat: string;
  rawRows: DataRow[];
  firstFiveRows: DataRow[];
  columns: ColumnProfile[];
  duplicateRowCount: number;
  totalMissingCells: number;
  completenessPct: number;
  qualityIssues: QualityIssue[];
  unstructuredWarning?: string;
}

export interface BusinessContext {
  sector: string;
  recordUnit: string;
  businessQuestion: string;
  targetOutcome: string;
  horizon: string;
  controllableVars: string;
  constraints: string;
}

export interface GeminiContextAnalysis {
  problemSynthesis: string;
  analyticalFocus: "predictivo" | "explicativo" | "mixto" | string;
  suggestedTargets: Array<{
    column: string;
    suitability: "alta" | "media" | "no_apta_lineal" | string;
    reason: string;
  }>;
  predictorAssessments: Array<{
    column: string;
    tier: "Recomendada" | "Opcional" | "No recomendada" | string;
    controllable: boolean;
    reason: string;
  }>;
  keyConstraintsAndRisks: string[];
}

export type NumericImputationMethod =
  | "none"
  | "mean"
  | "median"
  | "group_median"
  | "knn";

export type CategoricalImputationMethod =
  | "none"
  | "mode"
  | "missing_category"
  | "group_mode";

export type TemporalImputationMethod =
  | "none"
  | "forward_fill"
  | "drop_invalid";

export type TargetMissingHandling = "drop_rows";

export type ImputationStrategy =
  | NumericImputationMethod
  | CategoricalImputationMethod
  | TemporalImputationMethod
  | TargetMissingHandling;

export interface CleaningColumnRule {
  column: string;
  recommendedStrategy: ImputationStrategy;
  selectedStrategy: ImputationStrategy;
  reason: string;
  groupByColumn?: string;
  standardizeCategories: boolean;
  parseDirtyNumbers: boolean;
  outlierAction: "keep_flagged" | "winsorize_iqr";
}

export interface SplitConfig {
  mode: "random" | "temporal";
  dateColumn?: string;
  trainPct: number;
  valPct: number;
  testPct: number;
  seed: number;
}

export interface CleaningChangeItem {
  column: string;
  action: string;
  affectedCount: number;
  detail: string;
}

export interface CleanedDatasetResult {
  allCleanRows: DataRow[];
  trainRows: DataRow[];
  valRows: DataRow[];
  testRows: DataRow[];
  droppedTargetRows: number;
  deduplicatedRows: number;
  completenessBefore: number;
  completenessAfter: number;
  changes: CleaningChangeItem[];
  learnedParametersNote: string;
  splitConfig: SplitConfig;
}

export type CorrelationMethod = "pearson" | "spearman" | "kendall";

export interface CorrelationMatrixResult {
  method: CorrelationMethod;
  columns: string[];
  matrix: number[][];
}

export interface CategoricalAssociation {
  colA: string;
  colB: string;
  metric: "Cramér's V" | "Eta-cuadrado (η²)";
  value: number;
  interpretation: string;
}

export interface VifEntry {
  column: string;
  vif: number;
  tolerance: number;
  severity: "normal" | "moderate" | "severe";
}

export type ModelMethod = "OLS_SIMPLE" | "OLS_MULTIPLE" | "RIDGE" | "LASSO";

export interface CoefficientStat {
  term: string;
  sourceColumn: string;
  isIntercept: boolean;
  isDummy: boolean;
  referenceCategory?: string;
  estimate: number;
  stdError: number | null;
  tStat: number | null;
  pValue: number | null;
  ciLow: number | null;
  ciHigh: number | null;
  vif: number | null;
  sign: "+" | "-" | "0";
}

export interface ResidualDiagnostics {
  durbinWatson: number;
  durbinWatsonNote: string;
  breuschPaganStat: number;
  breuschPaganPValue: number;
  breuschPaganNote: string;
  jarqueBeraStat: number;
  jarqueBeraPValue: number;
  jarqueBeraNote: string;
  residualsVsFitted: Array<{ fitted: number; residual: number; actual: number }>;
  qqPoints: Array<{ theoretical: number; sample: number }>;
}

export interface TrainedModel {
  id: string;
  name: string;
  method: ModelMethod;
  targetColumn: string;
  selectedPredictors: string[];
  encodedTerms: string[];
  equation: string;
  nTrain: number;
  nVal: number;
  nTest: number;
  r2Train: number;
  adjR2Train: number;
  rmseTrain: number;
  rmseVal: number;
  rmseTest: number;
  maeVal: number;
  maeTest: number;
  cvRmseMean: number;
  cvMethod: string;
  fStat: number | null;
  fPValue: number | null;
  aic: number | null;
  bic: number | null;
  maxVif: number;
  residualStdError: number;
  coefficients: CoefficientStat[];
  diagnostics: ResidualDiagnostics;
  regularizationAlpha?: number;
  inferenceValid: boolean;
  inferenceWarning?: string;
  signCoherenceNote: string;
  isReliable: boolean;
  reliabilityReasons: string[];
  // Needed for exact consistent prediction using train-fitted encoding & residual scale
  predictorSpecs: Array<{
    column: string;
    type: VariableType;
    categories?: string[];
    referenceCategory?: string;
    trainMean: number;
    trainStd: number;
    trainMin: number;
    trainMax: number;
  }>;
  xtxInverse?: number[][];
}

export type SelectionFocus = "predictive" | "explanatory" | "balanced";

export interface ModelComparisonRow {
  modelId: string;
  name: string;
  method: ModelMethod;
  adjR2: number;
  rmseVal: number;
  cvRmse: number;
  maeVal: number;
  significantCoefRatio: string;
  fPValue: number | null;
  maxVif: number;
  residualHealth: string;
  interpretability: "Alta" | "Media";
  priorityNotes: string;
}

export interface ModelSelectionResult {
  focus: SelectionFocus;
  recommendedModelId: string | null;
  noReliableModel: boolean;
  headline: string;
  rationale: string;
  advantages: string[];
  limitations: string[];
  contributingVariables: string[];
  attentionAssumptions: string[];
  comparisonRows: ModelComparisonRow[];
}

export type SimulationStrategyType =
  | "aux_ols_simple"
  | "aux_ols_multiple"
  | "time_trend"
  | "bootstrap"
  | "monte_carlo_normal"
  | "categorical_empirical"
  | "user_controlled";

export interface VariableSimulationPlan {
  column: string;
  type: VariableType;
  unit: string;
  controllable: boolean;
  historicalMean?: number;
  historicalMedian?: number;
  historicalStd?: number;
  historicalMin?: number;
  historicalMax?: number;
  trendSlope?: number;
  trendR2?: number;
  seasonalityDetected: boolean;
  correlatedWith: Array<{ column: string; r: number }>;
  categories?: Array<{ value: string; prob: number }>;
  recommendedStrategy: SimulationStrategyType;
  selectedStrategy: SimulationStrategyType;
  justification: string;
  auxFitQuality: string;
  auxR2?: number;
  auxDrivers?: string[];
  viableMin?: number;
  viableMax?: number;
  uncertaintySummary: string;
  availableStrategies: Array<{ value: SimulationStrategyType; label: string }>;
  sparkline: number[];
}

export interface SimulatedRowRecord {
  periodIndex: number;
  sourceType: "Simulado";
  values: Record<string, number | string>;
  predictedY: number;
  piLow: number;
  piHigh: number;
}

export type ScenarioPresetType =
  | "base"
  | "optimista"
  | "conservador"
  | "adverso"
  | "personalizado"
  | "probabilistico";

export interface ScenarioInputItem {
  column: string;
  type: VariableType;
  unit: string;
  baseValue: number | string;
  scenarioValue: number | string;
  generationMethod: SimulationStrategyType;
  historicalMin?: number;
  historicalMax?: number;
  outOfRange: boolean;
}

export interface SensitivityBar {
  column: string;
  impactPositive: number;
  impactNegative: number;
  absSpread: number;
}

export interface ScenarioResult {
  id: string;
  name: string;
  type: ScenarioPresetType;
  inputs: ScenarioInputItem[];
  predictedY: number;
  baseY: number;
  diffVsBase: number;
  pctVsBase: number;
  piLow: number;
  piHigh: number;
  uncertaintyHalfWidth: number;
  outOfRangeWarnings: string[];
  sensitivity: SensitivityBar[];
  monteCarloSummary?: {
    iterations: number;
    mean: number;
    std: number;
    p10: number;
    p50: number;
    p90: number;
    min: number;
    max: number;
    histogram: Array<{ bin: string; count: number }>;
  };
}

export interface GeminiScenarioRecommendation {
  scenarioId: string;
  scenarioName: string;
  hallazgo: string;
  interpretacion: string;
  recomendacion: string;
  impactoEsperado: string;
  riesgos: string;
  distincionCausal: string;
  nivelConfianza: string;
}

export interface GeminiExecutiveNarrative {
  executiveSummary: string;
  methodologicalNotes: string;
  actionPlan: string[];
}
