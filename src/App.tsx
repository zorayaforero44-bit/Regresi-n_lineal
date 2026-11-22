import React, { useCallback, useMemo, useState } from "react";
import { BookOpen, Menu, X } from "lucide-react";
import {
  BusinessContext,
  CleanedDatasetResult,
  CleaningColumnRule,
  ColumnProfile,
  DatasetState,
  GeminiContextAnalysis,
  GeminiExecutiveNarrative,
  GeminiScenarioRecommendation,
  ModelSelectionResult,
  SelectionFocus,
  SimulatedRowRecord,
  SimulationStrategyType,
  SplitConfig,
  TrainedModel,
  VariableSimulationPlan,
} from "./types/analytics";
import {
  getSampleDatasets,
  parseUploadedFile,
  profileDataset,
  SampleDatasetSpec,
} from "./engine/dataParser";
import { buildRecommendedCleaningRules, executeCleaningPipeline } from "./engine/cleaningPipeline";
import {
  buildCandidateModels,
  evaluateAndCompareModels,
} from "./engine/regressionEngine";
import {
  analyzeVariablesForSimulation,
  computeScenarioMatrix,
  generateSimulatedDataset,
} from "./engine/simulationEngine";
import { exportExecutivePdfReport } from "./engine/exportEngine";
import { MethodologyModal } from "./components/MethodologyModal";
import { StepBusinessContext, StepLoadAndDictionary } from "./components/steps/StepDataAndContext";
import { StepCleaningAndSplit, StepEda } from "./components/steps/StepCleanAndEda";
import {
  StepInterpretation,
  StepModeling,
  StepValidationComparator,
  StepVariableSelection,
} from "./components/steps/StepModelingAndValidation";
import {
  StepExecutiveDashboard,
  StepRecommendations,
  StepScenarioLab,
  StepSimulationEngine,
} from "./components/steps/StepSimulationAndDashboard";

type StepId =
  | "01_load"
  | "02_context"
  | "03_clean"
  | "04_eda"
  | "05_vars"
  | "06_model"
  | "07_compare"
  | "08_interpret"
  | "09_sim_engine"
  | "10_scenarios"
  | "11_recommendations"
  | "12_dashboard";

const WORKFLOW_STEPS: Array<{ id: StepId; num: string; label: string; group: string }> = [
  { id: "01_load", num: "01", label: "Carga y Diccionario", group: "Datos y Limpieza" },
  { id: "02_context", num: "02", label: "Contexto de Negocio", group: "Datos y Limpieza" },
  { id: "03_clean", num: "03", label: "Limpieza y Split", group: "Datos y Limpieza" },
  { id: "04_eda", num: "04", label: "Exploración (EDA)", group: "Exploración y Modelos" },
  { id: "05_vars", num: "05", label: "Selección de Variables", group: "Exploración y Modelos" },
  { id: "06_model", num: "06", label: "Modelado OLS / Reg.", group: "Exploración y Modelos" },
  { id: "07_compare", num: "07", label: "Comparación y Validación", group: "Exploración y Modelos" },
  { id: "08_interpret", num: "08", label: "Interpretación", group: "Exploración y Modelos" },
  { id: "09_sim_engine", num: "09", label: "Motor de Simulación", group: "Simulación y Escenarios" },
  { id: "10_scenarios", num: "10", label: "Laboratorio de Escenarios", group: "Simulación y Escenarios" },
  { id: "11_recommendations", num: "11", label: "Recomendaciones IA", group: "Simulación y Escenarios" },
  { id: "12_dashboard", num: "12", label: "Dashboard y Exportación", group: "Dashboard Ejecutivo" },
];

const EMPTY_CONTEXT: BusinessContext = {
  sector: "",
  recordUnit: "",
  businessQuestion: "",
  targetOutcome: "",
  horizon: "",
  controllableVars: "",
  constraints: "",
};

export default function App() {
  const samples = useMemo(() => getSampleDatasets(), []);
  const [activeStep, setActiveStep] = useState<StepId>("01_load");
  const [guidedMode, setGuidedMode] = useState<boolean>(true);
  const [methodologyOpen, setMethodologyOpen] = useState<boolean>(false);
  const [mobileNavOpen, setMobileNavOpen] = useState<boolean>(false);

  // Module 1 & 2 State
  const [dataset, setDataset] = useState<DatasetState | null>(null);
  const [colOverrides, setColOverrides] = useState<
    Record<string, Partial<Pick<ColumnProfile, "type" | "unit" | "description" | "businessRole">>>
  >({});
  const [businessContext, setBusinessContext] = useState<BusinessContext>(EMPTY_CONTEXT);
  const [geminiContextAnalysis, setGeminiContextAnalysis] = useState<GeminiContextAnalysis | null>(null);

  // Module 3 State
  const [targetColumn, setTargetColumn] = useState<string | null>(null);
  const [cleaningRules, setCleaningRules] = useState<CleaningColumnRule[]>([]);
  const [splitConfig, setSplitConfig] = useState<SplitConfig>({
    mode: "random",
    trainPct: 70,
    valPct: 15,
    testPct: 15,
    seed: 42,
  });
  const [deduplicate, setDeduplicate] = useState<boolean>(true);
  const [cleaned, setCleaned] = useState<CleanedDatasetResult | null>(null);

  // Module 5, 6, 7 State
  const [selectedPredictors, setSelectedPredictors] = useState<string[]>([]);
  const [includeRegularization, setIncludeRegularization] = useState<boolean>(true);
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [selectionFocus, setSelectionFocus] = useState<SelectionFocus>("balanced");
  const [modelSelection, setModelSelection] = useState<ModelSelectionResult | null>(null);

  // Module 9, 10, 11, 12 State
  const [simPlans, setSimPlans] = useState<VariableSimulationPlan[]>([]);
  const [horizonCount, setHorizonCount] = useState<number>(12);
  const [simRows, setSimRows] = useState<SimulatedRowRecord[]>([]);
  const [customOverrides, setCustomOverrides] = useState<Record<string, number | string>>({});
  const [recommendations, setRecommendations] = useState<Record<string, GeminiScenarioRecommendation>>({});
  const [narrative, setNarrative] = useState<GeminiExecutiveNarrative | null>(null);

  const initializeFromDataset = useCallback((ds: DatasetState, ctx?: BusinessContext) => {
    setDataset(ds);
    setColOverrides({});
    setBusinessContext(ctx ?? EMPTY_CONTEXT);
    setGeminiContextAnalysis(null);
    setCleaned(null);
    setModels([]);
    setActiveModelId(null);
    setModelSelection(null);
    setSimPlans([]);
    setSimRows([]);
    setCustomOverrides({});
    setRecommendations({});
    setNarrative(null);

    const numericCols = ds.columns.filter(
      (c) => (c.type === "numeric_continuous" || c.type === "numeric_discrete") && !c.isConstant
    );
    const defaultY = numericCols[numericCols.length - 1]?.name ?? ds.columns[0]?.name ?? null;
    setTargetColumn(defaultY);

    const dateCol = ds.columns.find((c) => c.type === "temporal")?.name;
    setSplitConfig({
      mode: dateCol ? "temporal" : "random",
      dateColumn: dateCol,
      trainPct: 70,
      valPct: 15,
      testPct: 15,
      seed: 42,
    });

    setCleaningRules(buildRecommendedCleaningRules(ds.columns, defaultY));

    const defaultXs = ds.columns
      .filter(
        (c) =>
          c.name !== defaultY &&
          !c.isConstant &&
          c.type !== "temporal" &&
          c.type !== "textual" &&
          c.businessRole !== "identifier_exclude"
      )
      .map((c) => c.name);
    setSelectedPredictors(defaultXs);
  }, []);

  const handleFileUpload = async (file: File) => {
    const ds = await parseUploadedFile(file);
    initializeFromDataset(ds);
  };

  const handleLoadSample = (sample: SampleDatasetSpec) => {
    const ds = profileDataset(`${sample.id}.csv`, "CSV", sample.rows);
    initializeFromDataset(ds, sample.defaultContext);
  };

  const handleUpdateColumn = (
    colName: string,
    patch: Partial<Pick<ColumnProfile, "type" | "unit" | "description" | "businessRole">>
  ) => {
    if (!dataset) return;
    const nextOverrides = {
      ...colOverrides,
      [colName]: { ...(colOverrides[colName] ?? {}), ...patch },
    };
    setColOverrides(nextOverrides);
    const updatedDs = profileDataset(dataset.fileName, dataset.fileFormat, dataset.rawRows, nextOverrides);
    setDataset(updatedDs);
    setCleaningRules(buildRecommendedCleaningRules(updatedDs.columns, targetColumn));
  };

  const handleAnalyzeContext = async () => {
    if (!dataset) return;
    const res = await fetch("/api/gemini/analyze-context", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        businessContext,
        rowCount: dataset.rawRows.length,
        columns: dataset.columns.map((c) => ({
          name: c.name,
          type: c.type,
          unit: c.unit,
          description: c.description,
          missingPct: Number(c.missingPct.toFixed(1)),
          uniqueCount: c.uniqueCount,
          mean: c.stats?.mean,
          min: c.stats?.min,
          max: c.stats?.max,
        })),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Error analizando contexto con Gemini.");
    setGeminiContextAnalysis(data);
  };

  const runCleaning = (overrideTarget?: string | null) => {
    if (!dataset) return;
    const tgt = overrideTarget !== undefined ? overrideTarget : targetColumn;
    const res = executeCleaningPipeline(dataset, cleaningRules, splitConfig, tgt, deduplicate);
    setCleaned(res);
    return res;
  };

  const rebuildSimulationForModel = useCallback(
    (cleanRes: CleanedDatasetResult, model: TrainedModel, ds: DatasetState, hCount: number) => {
      const colMap = new Map(ds.columns.map((c) => [c.name, c]));
      const dateCol = ds.columns.find((c) => c.type === "temporal")?.name;
      const plans = analyzeVariablesForSimulation(cleanRes.allCleanRows, model, colMap, dateCol);
      setSimPlans(plans);
      setCustomOverrides({});
      const generated = generateSimulatedDataset(cleanRes.allCleanRows, plans, model, hCount, 2026);
      setSimRows(generated);
    },
    []
  );

  const handleTrainModels = () => {
    if (!dataset || !cleaned || !targetColumn || selectedPredictors.length === 0) return;
    const colMap = new Map(dataset.columns.map((c) => [c.name, c]));
    const isTemporal = splitConfig.mode === "temporal";
    const trained = buildCandidateModels(
      cleaned.trainRows,
      cleaned.valRows,
      cleaned.testRows,
      targetColumn,
      selectedPredictors,
      colMap,
      isTemporal,
      includeRegularization
    );
    setModels(trained);
    const sel = evaluateAndCompareModels(trained, selectionFocus);
    setModelSelection(sel);
    const chosenId = sel.recommendedModelId ?? trained[0]?.id ?? null;
    setActiveModelId(chosenId);
    const chosenModel = trained.find((m) => m.id === chosenId) ?? trained[0];
    if (chosenModel) {
      rebuildSimulationForModel(cleaned, chosenModel, dataset, horizonCount);
    }
    setRecommendations({});
    setNarrative(null);
    setActiveStep("06_model");
  };

  const handleChangeSelectionFocus = (focus: SelectionFocus) => {
    setSelectionFocus(focus);
    if (models.length > 0) {
      const sel = evaluateAndCompareModels(models, focus);
      setModelSelection(sel);
      if (sel.recommendedModelId) {
        setActiveModelId(sel.recommendedModelId);
        const m = models.find((x) => x.id === sel.recommendedModelId);
        if (m && cleaned && dataset) rebuildSimulationForModel(cleaned, m, dataset, horizonCount);
      }
    }
  };

  const handleSelectActiveModel = (id: string) => {
    setActiveModelId(id);
    const m = models.find((x) => x.id === id);
    if (m && cleaned && dataset) {
      rebuildSimulationForModel(cleaned, m, dataset, horizonCount);
      setRecommendations({});
      setNarrative(null);
    }
  };

  const activeModel = useMemo(
    () => models.find((m) => m.id === activeModelId) ?? models[0] ?? null,
    [models, activeModelId]
  );

  const scenarios = useMemo(() => {
    if (!cleaned || !activeModel || simPlans.length === 0) return [];
    return computeScenarioMatrix(cleaned.allCleanRows, simPlans, activeModel, customOverrides, 1000, 777);
  }, [cleaned, activeModel, simPlans, customOverrides]);

  const handleUpdateSimStrategy = (col: string, strategy: SimulationStrategyType) => {
    const nextPlans = simPlans.map((p) => (p.column === col ? { ...p, selectedStrategy: strategy } : p));
    setSimPlans(nextPlans);
    if (cleaned && activeModel) {
      setSimRows(generateSimulatedDataset(cleaned.allCleanRows, nextPlans, activeModel, horizonCount, 2026));
    }
  };

  const handleGenerateRecommendation = async (scenarioId: string) => {
    if (!activeModel || !dataset) return;
    const sc = scenarios.find((s) => s.id === scenarioId);
    if (!sc) return;
    const targetUnit = dataset.columns.find((c) => c.name === activeModel.targetColumn)?.unit ?? "unidades";

    const res = await fetch("/api/gemini/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        businessQuestion: businessContext.businessQuestion,
        sector: businessContext.sector,
        constraints: businessContext.constraints,
        targetName: activeModel.targetColumn,
        targetUnit,
        modelSummary: {
          name: activeModel.name,
          method: activeModel.method,
          nObs: activeModel.nTrain,
          r2: activeModel.r2Train,
          adjR2: activeModel.adjR2Train,
          outOfSampleRmse: activeModel.rmseVal,
          outOfSampleMae: activeModel.maeVal,
          fStat: activeModel.fStat,
          fPValue: activeModel.fPValue,
          maxVif: activeModel.maxVif,
          isReliable: activeModel.isReliable,
          reliabilityNote: activeModel.reliabilityReasons.join("; ") || "Modelo estadísticamente válido",
          coefficients: activeModel.coefficients.map((c) => ({
            term: c.term,
            estimate: c.estimate,
            pValue: c.pValue,
            ciLow: c.ciLow,
            ciHigh: c.ciHigh,
            vif: c.vif,
          })),
        },
        scenario: {
          name: sc.name,
          type: sc.type,
          predictedY: sc.predictedY,
          baseY: sc.baseY,
          diffVsBase: sc.diffVsBase,
          pctVsBase: sc.pctVsBase,
          piLow: sc.piLow,
          piHigh: sc.piHigh,
          outOfRangeWarnings: sc.outOfRangeWarnings,
          inputs: sc.inputs.map((i) => ({
            variable: i.column,
            baseValue: i.baseValue,
            scenarioValue: i.scenarioValue,
            generationMethod: i.generationMethod,
            unit: i.unit,
          })),
        },
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Error generando recomendación.");
    setRecommendations((prev) => ({
      ...prev,
      [scenarioId]: {
        scenarioId,
        scenarioName: sc.name,
        ...data,
      },
    }));
  };

  const handleGenerateNarrative = async () => {
    if (!cleaned || !activeModel) return;
    const res = await fetch("/api/gemini/executive-summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        businessQuestion: businessContext.businessQuestion,
        sector: businessContext.sector,
        targetName: activeModel.targetColumn,
        dataQualitySummary: {
          rowsOriginal: dataset?.rawRows.length ?? 0,
          rowsClean: cleaned.allCleanRows.length,
          completenessBefore: cleaned.completenessBefore,
          completenessAfter: cleaned.completenessAfter,
          trainCount: cleaned.trainRows.length,
          valCount: cleaned.valRows.length,
          testCount: cleaned.testRows.length,
        },
        bestModel: {
          name: activeModel.name,
          method: activeModel.method,
          adjR2: activeModel.adjR2Train,
          outOfSampleRmse: activeModel.rmseVal,
          mae: activeModel.maeVal,
          fPValue: activeModel.fPValue,
          reliabilityNote: activeModel.reliabilityReasons.join("; ") || "Válido",
        },
        scenariosSummary: scenarios.map((s) => ({
          name: s.name,
          predictedY: s.predictedY,
          pctVsBase: s.pctVsBase,
        })),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Error generando narrativa ejecutiva.");
    setNarrative(data);
  };

  const handlePrimaryExportAction = () => {
    if (dataset && cleaned && activeModel && modelSelection) {
      exportExecutivePdfReport({
        datasetName: dataset.fileName,
        businessContext,
        cleaned,
        selectedModel: activeModel,
        selection: modelSelection,
        scenarios,
        recommendations,
        narrative,
      });
    } else {
      setActiveStep("12_dashboard");
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Top Bar Contract: Strict 3-zone layout (Brand title | 4 concise nav links | 1 primary CTA) */}
      <header className="sticky top-0 z-30 bg-white border-b border-slate-200 px-6 py-3.5 flex items-center justify-between gap-8">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveStep("01_load");
          }}
          className="text-lg font-bold tracking-tight text-[#0B1F3A] whitespace-nowrap shrink-0 font-display"
        >
          Linear Regression Intelligence
        </a>

        <nav className="hidden lg:flex items-center gap-6 text-sm font-medium text-slate-600">
          <button
            type="button"
            onClick={() => setActiveStep("01_load")}
            className={`whitespace-nowrap shrink-0 transition-colors hover:text-[#0B1F3A] ${
              ["01_load", "02_context", "03_clean"].includes(activeStep)
                ? "text-[#0B1F3A] font-semibold underline decoration-teal-600 decoration-2 underline-offset-8"
                : ""
            }`}
          >
            Datos y Limpieza
          </button>
          <button
            type="button"
            onClick={() => setActiveStep(models.length > 0 ? "06_model" : "04_eda")}
            className={`whitespace-nowrap shrink-0 transition-colors hover:text-[#0B1F3A] ${
              ["04_eda", "05_vars", "06_model", "07_compare", "08_interpret"].includes(activeStep)
                ? "text-[#0B1F3A] font-semibold underline decoration-teal-600 decoration-2 underline-offset-8"
                : ""
            }`}
          >
            Exploración y Modelos
          </button>
          <button
            type="button"
            onClick={() => setActiveStep(simPlans.length > 0 ? "10_scenarios" : "09_sim_engine")}
            className={`whitespace-nowrap shrink-0 transition-colors hover:text-[#0B1F3A] ${
              ["09_sim_engine", "10_scenarios", "11_recommendations"].includes(activeStep)
                ? "text-[#0B1F3A] font-semibold underline decoration-teal-600 decoration-2 underline-offset-8"
                : ""
            }`}
          >
            Simulación y Escenarios
          </button>
          <button
            type="button"
            onClick={() => setActiveStep("12_dashboard")}
            className={`whitespace-nowrap shrink-0 transition-colors hover:text-[#0B1F3A] ${
              activeStep === "12_dashboard"
                ? "text-[#0B1F3A] font-semibold underline decoration-teal-600 decoration-2 underline-offset-8"
                : ""
            }`}
          >
            Dashboard Ejecutivo
          </button>
        </nav>

        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => setMobileNavOpen((v) => !v)}
            className="lg:hidden p-2 rounded-lg border border-slate-200 text-slate-700"
            aria-label="Menú de módulos"
          >
            {mobileNavOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
          <button
            type="button"
            onClick={handlePrimaryExportAction}
            className="px-4 py-2 text-xs font-semibold text-white bg-[#0B1F3A] rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Exportar Informe
          </button>
        </div>
      </header>

      {/* Workspace Sub-Toolbar (Personal Brand + Guided/Advanced Mode + Academic Credits) */}
      <div className="bg-white border-b border-slate-200 px-6 py-2.5 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex flex-wrap items-center gap-2 text-slate-600">
          <span className="font-semibold text-[#0B1F3A]">Linear Regression Intelligence — By Zoraya Forero</span>
          <span aria-hidden="true" className="text-slate-300">·</span>
          <span>Predictive Analytics &amp; Business Decision Simulator</span>
          {dataset && (
            <>
              <span aria-hidden="true" className="text-slate-300">·</span>
              <span className="font-mono text-teal-700 font-medium">{dataset.fileName}</span>
            </>
          )}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded-lg">
            <button
              type="button"
              onClick={() => setGuidedMode(true)}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                guidedMode ? "bg-white text-[#0B1F3A] shadow-xs" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Modo Guiado
            </button>
            <button
              type="button"
              onClick={() => setGuidedMode(false)}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                !guidedMode ? "bg-white text-[#0B1F3A] shadow-xs" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Modo Avanzado
            </button>
          </div>
          <button
            type="button"
            onClick={() => setMethodologyOpen(true)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-slate-200 rounded-lg text-xs font-medium text-slate-700 hover:bg-slate-50 whitespace-nowrap"
          >
            <BookOpen className="w-3.5 h-3.5 text-teal-600" />
            Fundamento Académico
          </button>
        </div>
      </div>

      {/* Main Workspace: Sidebar + Content Viewport */}
      <div className="flex-1 flex flex-col lg:flex-row max-w-[1600px] w-full mx-auto">
        {/* Left Sidebar */}
        <aside
          className={`${
            mobileNavOpen ? "block" : "hidden"
          } lg:block w-full lg:w-[265px] shrink-0 bg-white border-r border-slate-200 p-4 space-y-5`}
        >
          <div className="space-y-1">
            <p className="px-2.5 text-[11px] font-semibold text-slate-400">Flujo Analítico Integral</p>
            <nav className="space-y-0.5">
              {WORKFLOW_STEPS.map((st) => {
                const isActive = st.id === activeStep;
                return (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => {
                      setActiveStep(st.id);
                      setMobileNavOpen(false);
                    }}
                    className={`w-full text-left px-2.5 py-2 rounded-lg text-xs flex items-center gap-2.5 transition-colors ${
                      isActive
                        ? "bg-[#0B1F3A] text-white font-semibold"
                        : "text-slate-700 hover:bg-slate-100"
                    }`}
                  >
                    <span
                      className={`font-mono tabular-nums text-[11px] ${
                        isActive ? "text-teal-300" : "text-slate-400"
                      }`}
                    >
                      {st.num}.
                    </span>
                    <span className="truncate">{st.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          <div className="border-t border-slate-200 pt-4 px-2 space-y-1.5 text-xs">
            <p className="font-bold text-[#0B1F3A]">Zoraya Forero</p>
            <p className="text-slate-500 leading-relaxed">
              Economista | Análisis de datos y automatización de procesos | Análisis financiero | Reportes para la toma
              de decisiones.
            </p>
          </div>
        </aside>

        {/* Active Module Content */}
        <main className="flex-1 p-5 md:p-8 min-w-0">
          {activeStep === "01_load" && (
            <StepLoadAndDictionary
              dataset={dataset}
              samples={samples}
              guidedMode={guidedMode}
              onFileUpload={handleFileUpload}
              onLoadSample={handleLoadSample}
              onUpdateColumn={handleUpdateColumn}
              onNext={() => setActiveStep("02_context")}
            />
          )}

          {activeStep === "02_context" && (
            <StepBusinessContext
              dataset={dataset}
              context={businessContext}
              analysis={geminiContextAnalysis}
              guidedMode={guidedMode}
              onChangeContext={(patch) => setBusinessContext((prev) => ({ ...prev, ...patch }))}
              onAnalyzeWithGemini={handleAnalyzeContext}
              onNext={() => setActiveStep("03_clean")}
            />
          )}

          {activeStep === "03_clean" && (
            <StepCleaningAndSplit
              dataset={dataset}
              targetColumn={targetColumn}
              rules={cleaningRules}
              splitConfig={splitConfig}
              deduplicate={deduplicate}
              cleaned={cleaned}
              guidedMode={guidedMode}
              onChangeTargetPreview={(col) => {
                setTargetColumn(col);
                if (dataset) setCleaningRules(buildRecommendedCleaningRules(dataset.columns, col));
              }}
              onUpdateRule={(col, patch) =>
                setCleaningRules((prev) => prev.map((r) => (r.column === col ? { ...r, ...patch } : r)))
              }
              onChangeSplit={(patch) => setSplitConfig((prev) => ({ ...prev, ...patch }))}
              onToggleDeduplicate={setDeduplicate}
              onExecuteCleaning={() => runCleaning()}
              onNext={() => setActiveStep("04_eda")}
            />
          )}

          {activeStep === "04_eda" && (
            <StepEda
              dataset={dataset}
              cleaned={cleaned}
              targetColumn={targetColumn}
              constraintsText={businessContext.constraints}
              guidedMode={guidedMode}
              onNext={() => setActiveStep("05_vars")}
            />
          )}

          {activeStep === "05_vars" && (
            <StepVariableSelection
              dataset={dataset}
              cleaned={cleaned}
              targetColumn={targetColumn}
              selectedPredictors={selectedPredictors}
              includeRegularization={includeRegularization}
              geminiAnalysis={geminiContextAnalysis}
              guidedMode={guidedMode}
              onSelectTarget={(col) => {
                setTargetColumn(col);
                setSelectedPredictors((prev) => prev.filter((p) => p !== col));
                if (dataset) {
                  const nextRules = buildRecommendedCleaningRules(dataset.columns, col);
                  setCleaningRules(nextRules);
                  setCleaned(executeCleaningPipeline(dataset, nextRules, splitConfig, col, deduplicate));
                }
              }}
              onTogglePredictor={(col) =>
                setSelectedPredictors((prev) =>
                  prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
                )
              }
              onToggleRegularization={setIncludeRegularization}
              onTrainModels={handleTrainModels}
            />
          )}

          {activeStep === "06_model" && (
            <StepModeling
              models={models}
              activeModelId={activeModelId}
              guidedMode={guidedMode}
              onSelectActiveModel={handleSelectActiveModel}
              onNext={() => setActiveStep("07_compare")}
            />
          )}

          {activeStep === "07_compare" && (
            <StepValidationComparator
              models={models}
              selection={modelSelection}
              focus={selectionFocus}
              activeModelId={activeModelId}
              guidedMode={guidedMode}
              onChangeFocus={handleChangeSelectionFocus}
              onSelectModel={handleSelectActiveModel}
              onNext={() => setActiveStep("08_interpret")}
            />
          )}

          {activeStep === "08_interpret" && (
            <StepInterpretation
              dataset={dataset}
              model={activeModel}
              guidedMode={guidedMode}
              onNext={() => setActiveStep("09_sim_engine")}
            />
          )}

          {activeStep === "09_sim_engine" && (
            <StepSimulationEngine
              plans={simPlans}
              simRows={simRows}
              model={activeModel}
              horizonCount={horizonCount}
              guidedMode={guidedMode}
              onUpdateStrategy={handleUpdateSimStrategy}
              onChangeHorizon={setHorizonCount}
              onGenerateSimulated={() => {
                if (cleaned && activeModel) {
                  setSimRows(
                    generateSimulatedDataset(cleaned.allCleanRows, simPlans, activeModel, horizonCount, 2026)
                  );
                }
              }}
              onNext={() => setActiveStep("10_scenarios")}
            />
          )}

          {activeStep === "10_scenarios" && (
            <StepScenarioLab
              dataset={dataset}
              model={activeModel}
              plans={simPlans}
              scenarios={scenarios}
              customOverrides={customOverrides}
              guidedMode={guidedMode}
              onChangeCustomInput={(col, val) =>
                setCustomOverrides((prev) => ({ ...prev, [col]: val }))
              }
              onResetCustom={() => setCustomOverrides({})}
              onNext={() => setActiveStep("11_recommendations")}
            />
          )}

          {activeStep === "11_recommendations" && (
            <StepRecommendations
              scenarios={scenarios}
              recommendations={recommendations}
              guidedMode={guidedMode}
              onGenerateRecommendation={handleGenerateRecommendation}
              onNext={() => setActiveStep("12_dashboard")}
            />
          )}

          {activeStep === "12_dashboard" && (
            <StepExecutiveDashboard
              dataset={dataset}
              businessContext={businessContext}
              cleaned={cleaned}
              models={models}
              selectedModel={activeModel}
              selection={modelSelection}
              simPlans={simPlans}
              simRows={simRows}
              scenarios={scenarios}
              recommendations={recommendations}
              narrative={narrative}
              onGenerateNarrative={handleGenerateNarrative}
            />
          )}
        </main>
      </div>

      {/* Footer with Personal Brand & Academic Credit */}
      <footer className="bg-white border-t border-slate-200 px-6 py-5 text-xs text-slate-600">
        <div className="max-w-[1600px] mx-auto flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <p className="font-semibold text-[#0B1F3A]">
              Linear Regression Intelligence — By Zoraya Forero
            </p>
            <p className="text-slate-500 mt-0.5">
              Economista | Análisis de datos y automatización de procesos | Análisis financiero | Reportes para la toma de decisiones.
            </p>
          </div>
          <p className="text-slate-500 max-w-2xl leading-relaxed">
            Desarrollado a partir de conceptos académicos de regresión y pronósticos empresariales abordados en clase
            por el profesor Hugo Alexander Martínez Gallego, complementados con documentación estadística y metodologías
            de analítica de datos.
          </p>
        </div>
      </footer>

      <MethodologyModal open={methodologyOpen} onClose={() => setMethodologyOpen(false)} />
    </div>
  );
}
