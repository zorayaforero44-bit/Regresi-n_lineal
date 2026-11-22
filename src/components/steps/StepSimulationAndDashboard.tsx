import React, { useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Download,
  FileSpreadsheet,
  FileText,
  Play,
  Sliders,
  Sparkles,
} from "lucide-react";
import {
  BusinessContext,
  CleanedDatasetResult,
  DatasetState,
  GeminiExecutiveNarrative,
  GeminiScenarioRecommendation,
  ModelSelectionResult,
  ScenarioResult,
  SimulatedRowRecord,
  SimulationStrategyType,
  TrainedModel,
  VariableSimulationPlan,
} from "../../types/analytics";
import { formatNum, formatPValue } from "../../engine/mathStats";
import {
  HistogramChart,
  ScenarioComparisonChart,
  SensitivityTornadoChart,
} from "../Charts";
import {
  exportExecutivePdfReport,
  exportFullExcelWorkbook,
  exportRowsToCsv,
  exportScenariosToCsv,
} from "../../engine/exportEngine";

export const StepSimulationEngine: React.FC<{
  plans: VariableSimulationPlan[];
  simRows: SimulatedRowRecord[];
  model: TrainedModel | null;
  horizonCount: number;
  guidedMode: boolean;
  onUpdateStrategy: (col: string, strategy: SimulationStrategyType) => void;
  onChangeHorizon: (n: number) => void;
  onGenerateSimulated: () => void;
  onNext: () => void;
}> = ({
  plans,
  simRows,
  model,
  horizonCount,
  guidedMode,
  onUpdateStrategy,
  onChangeHorizon,
  onGenerateSimulated,
  onNext,
}) => {
  if (!model || plans.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Entrena y selecciona un modelo antes de configurar la simulación individual.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">09. Motor Estadístico Individual de Simulación</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Evaluación y Estrategia de Generación por Variable Independiente
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Antes de construir escenarios, evaluamos cada predictor individualmente para elegir el mecanismo más
              plausible (regresión auxiliar, tendencia temporal, bootstrap, Monte Carlo, muestreo categórico o control
              gerencial). Los datos simulados nunca se mezclan con el entrenamiento.
            </p>
          )}
        </div>
        <button
          onClick={onNext}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
        >
          Ir al Laboratorio de Escenarios
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Controls Bar */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <label className="flex items-center gap-2 font-medium text-slate-700">
            Horizonte de registros simulados:
            <input
              type="number"
              min={6}
              max={60}
              value={horizonCount}
              onChange={(e) => onChangeHorizon(Math.max(6, Math.min(60, Number(e.target.value) || 12)))}
              className="w-16 border border-slate-200 rounded-md px-2 py-1 font-mono text-center"
            />
          </label>
          <span className="text-slate-400">·</span>
          <span className="text-slate-600">
            Registros marcados explícitamente como <strong>"Simulado"</strong> (separados de reales e imputados).
          </span>
        </div>
        <button
          type="button"
          onClick={onGenerateSimulated}
          className="inline-flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap"
        >
          <Play className="w-3.5 h-3.5" />
          Generar Valores Simulados ({horizonCount})
        </button>
      </div>

      {/* Individual Variable Simulation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {plans.map((p) => {
          const maxSpark = Math.max(...p.sparkline, 1);
          const minSpark = Math.min(...p.sparkline, 0);
          const spanSpark = maxSpark - minSpark || 1;

          return (
            <div key={p.column} className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
              <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-3">
                <div>
                  <p className="text-xs text-slate-500">
                    {p.type} · {p.unit} · {p.controllable ? "Controlable por el negocio" : "Externa / Condicionada"}
                  </p>
                  <h3 className="text-base font-bold text-[#0B1F3A] mt-0.5">{p.column}</h3>
                </div>
                {/* Mini SVG Sparkline */}
                {p.sparkline.length > 1 && (
                  <svg viewBox="0 0 90 28" className="w-24 h-7 shrink-0">
                    <polyline
                      fill="none"
                      stroke="#0D9488"
                      strokeWidth="1.8"
                      points={p.sparkline
                        .map((v, i) => {
                          const x = (i / (p.sparkline.length - 1)) * 86 + 2;
                          const y = 25 - ((v - minSpark) / spanSpark) * 22;
                          return `${x},${y}`;
                        })
                        .join(" ")}
                    />
                  </svg>
                )}
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                  <span className="text-slate-500 block">Calidad / Ajuste Auxiliar</span>
                  <span className="font-mono font-semibold text-[#0B1F3A]">{p.auxFitQuality}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                  <span className="text-slate-500 block">Incertidumbre Histórica</span>
                  <span className="font-mono text-slate-700">{p.uncertaintySummary}</span>
                </div>
              </div>

              {p.correlatedWith.length > 0 && (
                <p className="text-xs text-slate-600">
                  <strong>Dependencias observadas:</strong>{" "}
                  {p.correlatedWith.map((c) => `${c.column} (r=${c.r.toFixed(2)})`).join(", ")}
                </p>
              )}

              <p className="text-xs text-slate-700 leading-relaxed">
                <strong>Justificación estadística:</strong> {p.justification}
              </p>

              <div className="pt-2 border-t border-slate-100">
                <label className="block text-[11px] font-semibold text-slate-600 mb-1">
                  Método de Simulación Activo (Recomendado: {p.recommendedStrategy})
                </label>
                <select
                  value={p.selectedStrategy}
                  onChange={(e) => onUpdateStrategy(p.column, e.target.value as SimulationStrategyType)}
                  className="w-full border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs bg-white font-medium text-[#0B1F3A]"
                >
                  {p.availableStrategies.map((st) => (
                    <option key={st.value} value={st.value}>
                      {st.label} {st.value === p.recommendedStrategy ? "(Recomendado)" : ""}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          );
        })}
      </div>

      {/* Generated Simulated Rows Preview */}
      {simRows.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-[#0B1F3A]">
                Muestra de Datos Simulados Generados ({simRows.length} periodos)
              </h3>
              <p className="text-xs text-slate-500">
                Calculados respetando dependencias auxiliares y alimentando el modelo seleccionado ({model.name}).
              </p>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                <tr>
                  <th className="py-2 px-3 font-medium">Periodo</th>
                  <th className="py-2 px-3 font-medium">Origen</th>
                  {plans.map((p) => (
                    <th key={p.column} className="py-2 px-3 font-medium text-right whitespace-nowrap">
                      {p.column}
                    </th>
                  ))}
                  <th className="py-2 px-3 font-medium text-right">Ŷ Estimado ({model.targetColumn})</th>
                  <th className="py-2 px-3 font-medium text-right">IC 95% Predicción</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {simRows.slice(0, 10).map((r) => (
                  <tr key={r.periodIndex} className="hover:bg-slate-50/50">
                    <td className="py-2 px-3 font-mono tabular-nums text-slate-500">T+{r.periodIndex}</td>
                    <td className="py-2 px-3 font-medium text-teal-700">{r.sourceType}</td>
                    {plans.map((p) => (
                      <td key={p.column} className="py-2 px-3 text-right font-mono tabular-nums">
                        {typeof r.values[p.column] === "number"
                          ? formatNum(r.values[p.column] as number, 2)
                          : String(r.values[p.column])}
                      </td>
                    ))}
                    <td className="py-2 px-3 text-right font-mono tabular-nums font-semibold text-[#0B1F3A]">
                      {formatNum(r.predictedY, 2)}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-600">
                      [{formatNum(r.piLow, 1)}, {formatNum(r.piHigh, 1)}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export const StepScenarioLab: React.FC<{
  dataset: DatasetState | null;
  model: TrainedModel | null;
  plans: VariableSimulationPlan[];
  scenarios: ScenarioResult[];
  customOverrides: Record<string, number | string>;
  guidedMode: boolean;
  onChangeCustomInput: (col: string, val: number | string) => void;
  onResetCustom: () => void;
  onNext: () => void;
}> = ({
  dataset,
  model,
  plans,
  scenarios,
  customOverrides,
  guidedMode,
  onChangeCustomInput,
  onResetCustom,
  onNext,
}) => {
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>("personalizado");

  if (!model || !dataset || scenarios.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Completa el modelo y el motor de simulación en los pasos anteriores.</p>
      </div>
    );
  }

  const targetUnit = dataset.columns.find((c) => c.name === model.targetColumn)?.unit ?? "unidades";
  const activeScenario = scenarios.find((s) => s.id === selectedScenarioId) ?? scenarios[4] ?? scenarios[0];
  const mcScenario = scenarios.find((s) => s.type === "probabilistico");

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">10. Laboratorio de Escenarios</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Simulador Interactivo de Decisiones Empresariales
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Evalúa los 6 escenarios obligatorios (Base, Optimista, Conservador, Adverso, Personalizado y
              Probabilístico Monte Carlo). El laboratorio respeta signos estimados, dependencias y emite alertas si
              extrapolas fuera del rango histórico.
            </p>
          )}
        </div>
        <button
          onClick={onNext}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
        >
          Ver Recomendaciones IA
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* 6 Mandatory Scenarios Cards Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {scenarios.map((sc) => {
          const active = sc.id === activeScenario.id;
          return (
            <button
              key={sc.id}
              type="button"
              onClick={() => setSelectedScenarioId(sc.id)}
              className={`text-left p-3.5 rounded-xl border transition-colors ${
                active ? "border-teal-600 bg-teal-50/50" : "border-slate-200 bg-white hover:border-slate-300"
              }`}
            >
              <p className="text-xs font-semibold text-[#0B1F3A] truncate">{sc.name}</p>
              <p className="text-lg font-bold font-mono tabular-nums text-[#0B1F3A] mt-1">
                {formatNum(sc.predictedY, 1)}
              </p>
              <p
                className={`text-xs font-mono tabular-nums mt-0.5 ${
                  sc.diffVsBase > 0 ? "text-teal-700" : sc.diffVsBase < 0 ? "text-red-700" : "text-slate-500"
                }`}
              >
                {sc.type === "base"
                  ? "Referencia Base"
                  : `${sc.pctVsBase >= 0 ? "+" : ""}${sc.pctVsBase.toFixed(1)}% vs Base`}
              </p>
            </button>
          );
        })}
      </div>

      {/* Custom Sliders + Scenario Comparison Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Interactive Controls for Personalizado */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-teal-600" />
              <h3 className="text-sm font-semibold text-[#0B1F3A]">Controles del Escenario Personalizado</h3>
            </div>
            <button
              type="button"
              onClick={() => {
                onResetCustom();
                setSelectedScenarioId("personalizado");
              }}
              className="text-xs font-medium text-teal-700 hover:underline"
            >
              Restaurar Base
            </button>
          </div>

          <div className="space-y-4 max-h-[430px] overflow-y-auto pr-1">
            {plans.map((p) => {
              const baseVal = p.categories ? p.categories[0]?.value ?? "" : p.historicalMedian ?? p.historicalMean ?? 0;
              const currentVal = customOverrides[p.column] ?? baseVal;

              if (p.categories && p.categories.length > 0) {
                return (
                  <div key={p.column} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-semibold text-slate-800">{p.column}</span>
                      <span className="text-slate-500">{p.controllable ? "Palanca" : "Categórica"}</span>
                    </div>
                    <select
                      value={String(currentVal)}
                      onChange={(e) => {
                        onChangeCustomInput(p.column, e.target.value);
                        setSelectedScenarioId("personalizado");
                      }}
                      className="w-full border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs bg-white"
                    >
                      {p.categories.map((c) => (
                        <option key={c.value} value={c.value}>
                          {c.value} ({(c.prob * 100).toFixed(0)}%)
                        </option>
                      ))}
                    </select>
                  </div>
                );
              }

              const hMin = p.historicalMin ?? 0;
              const hMax = p.historicalMax ?? 100;
              const span = hMax - hMin || 10;
              const sliderMin = Number((hMin - span * 0.2).toFixed(2));
              const sliderMax = Number((hMax + span * 0.2).toFixed(2));
              const step = p.type === "numeric_discrete" || p.type === "binary" ? 1 : Number((span / 100).toFixed(2)) || 0.1;
              const numVal = Number(currentVal);
              const isOut = numVal < hMin || numVal > hMax;

              return (
                <div key={p.column} className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-800">{p.column}</span>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        step={step}
                        value={numVal}
                        onChange={(e) => {
                          onChangeCustomInput(p.column, Number(e.target.value));
                          setSelectedScenarioId("personalizado");
                        }}
                        className="w-24 border border-slate-200 rounded px-2 py-0.5 text-right font-mono tabular-nums text-xs"
                      />
                    </div>
                  </div>
                  <input
                    type="range"
                    min={sliderMin}
                    max={sliderMax}
                    step={step}
                    value={Math.max(sliderMin, Math.min(sliderMax, numVal))}
                    onChange={(e) => {
                      onChangeCustomInput(p.column, Number(e.target.value));
                      setSelectedScenarioId("personalizado");
                    }}
                    className="w-full exec-slider"
                  />
                  <div className="flex justify-between text-[11px] font-mono text-slate-400">
                    <span>Histórico: [{formatNum(hMin, 1)}, {formatNum(hMax, 1)}]</span>
                    {isOut && <span className="text-amber-700 font-sans font-medium">Fuera de rango histórico</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Comparison Chart + Tornado Sensitivity */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-[#0B1F3A]">
              Comparación de Resultados Estimados ({model.targetColumn} · {targetUnit})
            </h3>
            <ScenarioComparisonChart scenarios={scenarios} targetUnit={targetUnit} />
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-[#0B1F3A]">
                  Análisis de Sensibilidad (Tornado ±0.5σ) — {activeScenario.name}
                </h3>
                <p className="text-xs text-slate-500">
                  Impacto marginal estimado en {model.targetColumn} al variar cada predictor individualmente.
                </p>
              </div>
            </div>
            <SensitivityTornadoChart bars={activeScenario.sensitivity} />
          </div>
        </div>
      </div>

      {/* Selected Scenario Detail + Out-of-range Warnings */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-[#0B1F3A]">
              Detalle del Escenario: {activeScenario.name}
            </h3>
            <p className="text-xs text-slate-500">
              Estimación puntual Ŷ = <strong className="font-mono">{formatNum(activeScenario.predictedY, 2)}</strong> {targetUnit} ·
              Intervalo de Predicción 95%: <strong className="font-mono">[{formatNum(activeScenario.piLow, 2)}, {formatNum(activeScenario.piHigh, 2)}]</strong> ·
              Diferencia vs Base: <strong className="font-mono">{activeScenario.diffVsBase >= 0 ? "+" : ""}{formatNum(activeScenario.diffVsBase, 2)} ({activeScenario.pctVsBase.toFixed(1)}%)</strong>
            </p>
          </div>
        </div>

        {activeScenario.outOfRangeWarnings.length > 0 && (
          <div className="p-3.5 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 space-y-1">
            <div className="flex items-center gap-2 font-semibold">
              <AlertTriangle className="w-4 h-4 text-amber-600" />
              <span>Advertencias de Extrapolación Fuera de Rango</span>
            </div>
            <ul className="list-disc pl-5 space-y-0.5">
              {activeScenario.outOfRangeWarnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="py-2 px-3 font-medium">Predictor</th>
                <th className="py-2 px-3 font-medium text-right">Valor Base</th>
                <th className="py-2 px-3 font-medium text-right">Valor Escenario</th>
                <th className="py-2 px-3 font-medium">Método de Generación</th>
                <th className="py-2 px-3 font-medium">Estado Rango</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {activeScenario.inputs.map((inp) => (
                <tr key={inp.column}>
                  <td className="py-2 px-3 font-semibold text-[#0B1F3A]">{inp.column}</td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-500">
                    {typeof inp.baseValue === "number" ? formatNum(inp.baseValue, 2) : inp.baseValue}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums font-semibold text-[#0B1F3A]">
                    {typeof inp.scenarioValue === "number" ? formatNum(inp.scenarioValue, 2) : inp.scenarioValue}
                  </td>
                  <td className="py-2 px-3 text-slate-600">{inp.generationMethod}</td>
                  <td className="py-2 px-3">
                    <span className={inp.outOfRange ? "text-amber-700 font-semibold" : "text-teal-700"}>
                      {inp.outOfRange ? "Fuera de rango histórico" : "Dentro de rango"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Monte Carlo Probabilistic Distribution */}
      {mcScenario?.monteCarloSummary && (
        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-[#0B1F3A]">
                Distribución Probabilística Monte Carlo ({mcScenario.monteCarloSummary.iterations} simulaciones)
              </h3>
              <p className="text-xs text-slate-500">
                Integra variabilidad conjunta de predictores e incertidumbre residual del modelo ({model.residualStdError.toFixed(2)}).
              </p>
            </div>
            <div className="flex flex-wrap gap-4 text-xs font-mono tabular-nums">
              <span>P10 (Pesimista): <strong>{formatNum(mcScenario.monteCarloSummary.p10, 1)}</strong></span>
              <span>P50 (Mediana): <strong>{formatNum(mcScenario.monteCarloSummary.p50, 1)}</strong></span>
              <span>P90 (Optimista): <strong className="text-teal-700">{formatNum(mcScenario.monteCarloSummary.p90, 1)}</strong></span>
            </div>
          </div>
          <HistogramChart
            data={mcScenario.monteCarloSummary.histogram.map((b) => ({ label: b.bin, count: b.count }))}
            color="#0B1F3A"
            height={210}
          />
        </div>
      )}
    </div>
  );
};

export const StepRecommendations: React.FC<{
  scenarios: ScenarioResult[];
  recommendations: Record<string, GeminiScenarioRecommendation>;
  guidedMode: boolean;
  onGenerateRecommendation: (scenarioId: string) => Promise<void>;
  onNext: () => void;
}> = ({ scenarios, recommendations, guidedMode, onGenerateRecommendation, onNext }) => {
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (id: string) => {
    setLoadingId(id);
    setError(null);
    try {
      await onGenerateRecommendation(id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error generando recomendación con Gemini.");
    } finally {
      setLoadingId(null);
    }
  };

  if (scenarios.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Ejecuta primero la simulación de escenarios en el paso 10.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">11. Recomendaciones Automáticas por Escenario</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Asesoría Estratégica Basada en Resultados Calculados (Gemini)
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Genera recomendaciones ejecutivas estructuradas (Hallazgo, Interpretación, Recomendación, Impacto esperado y
              Riesgos) apoyadas exclusivamente en los coeficientes, métricas e intervalos calculados.
            </p>
          )}
        </div>
        <button
          onClick={onNext}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
        >
          Ir al Dashboard Ejecutivo
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {error && (
        <div className="p-3.5 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700">{error}</div>
      )}

      <div className="space-y-4">
        {scenarios.map((sc) => {
          const rec = recommendations[sc.id];
          const isBusy = loadingId === sc.id;
          return (
            <div key={sc.id} className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
                <div>
                  <h3 className="text-base font-bold text-[#0B1F3A]">{sc.name}</h3>
                  <p className="text-xs font-mono text-slate-600 tabular-nums mt-0.5">
                    Ŷ = {formatNum(sc.predictedY, 2)} · Variación vs Base: {sc.pctVsBase >= 0 ? "+" : ""}
                    {sc.pctVsBase.toFixed(1)}% · IC [{formatNum(sc.piLow, 1)}, {formatNum(sc.piHigh, 1)}]
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => handleGenerate(sc.id)}
                  disabled={isBusy}
                  className="inline-flex items-center gap-2 px-3.5 py-2 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg transition-colors disabled:opacity-50 whitespace-nowrap"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  {isBusy ? "Analizando escenario..." : rec ? "Regenerar recomendación" : "Generar recomendación IA"}
                </button>
              </div>

              {!rec ? (
                <p className="text-xs text-slate-500">
                  Haz clic en "Generar recomendación IA" para obtener el dictamen estratégico de este escenario basado en tus métricas reales.
                </p>
              ) : (
                <div className="space-y-3 text-xs leading-relaxed">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                      <p className="font-bold text-[#0B1F3A] mb-1">Hallazgo</p>
                      <p className="text-slate-700">{rec.hallazgo}</p>
                    </div>
                    <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                      <p className="font-bold text-[#0B1F3A] mb-1">Interpretación</p>
                      <p className="text-slate-700">{rec.interpretacion}</p>
                    </div>
                    <div className="p-3.5 rounded-lg bg-teal-50/50 border border-teal-200">
                      <p className="font-bold text-teal-900 mb-1">Recomendación</p>
                      <p className="text-slate-800">{rec.recomendacion}</p>
                    </div>
                    <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
                      <p className="font-bold text-[#0B1F3A] mb-1">Impacto Esperado</p>
                      <p className="text-slate-700">{rec.impactoEsperado}</p>
                    </div>
                  </div>
                  <div className="p-3.5 rounded-lg bg-amber-50/70 border border-amber-200 space-y-1">
                    <p className="font-bold text-amber-950">Riesgos y Restricciones ({rec.nivelConfianza})</p>
                    <p className="text-amber-900">{rec.riesgos}</p>
                    <p className="text-slate-700 pt-1">
                      <strong>Predicción vs Asociación vs Causalidad:</strong> {rec.distincionCausal}
                    </p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const StepExecutiveDashboard: React.FC<{
  dataset: DatasetState | null;
  businessContext: BusinessContext;
  cleaned: CleanedDatasetResult | null;
  models: TrainedModel[];
  selectedModel: TrainedModel | null;
  selection: ModelSelectionResult | null;
  simPlans: VariableSimulationPlan[];
  simRows: SimulatedRowRecord[];
  scenarios: ScenarioResult[];
  recommendations: Record<string, GeminiScenarioRecommendation>;
  narrative: GeminiExecutiveNarrative | null;
  onGenerateNarrative: () => Promise<void>;
}> = ({
  dataset,
  businessContext,
  cleaned,
  models,
  selectedModel,
  selection,
  simPlans,
  simRows,
  scenarios,
  recommendations,
  narrative,
  onGenerateNarrative,
}) => {
  const [genBusy, setGenBusy] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);

  if (!dataset || !cleaned || !selectedModel || !selection) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">
          Completa el flujo de limpieza, modelado y simulación para visualizar el Dashboard Ejecutivo.
        </p>
      </div>
    );
  }

  const targetUnit = dataset.columns.find((c) => c.name === selectedModel.targetColumn)?.unit ?? "unidades";
  const baseScenario = scenarios.find((s) => s.type === "base");
  const optScenario = scenarios.find((s) => s.type === "optimista");

  const runNarrative = async () => {
    setGenBusy(true);
    setGenError(null);
    try {
      await onGenerateNarrative();
    } catch (e: unknown) {
      setGenError(e instanceof Error ? e.message : "Error generando resumen ejecutivo.");
    } finally {
      setGenBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">12. Dashboard Ejecutivo y Exportación</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Resumen Gerencial de Analítica Predictiva y Simulación
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            {businessContext.sector || "Sector Empresarial"} · Pregunta: {businessContext.businessQuestion || "Análisis de regresión"}
          </p>
        </div>

        {/* Real Export Buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => exportRowsToCsv(cleaned.allCleanRows, `Base_Limpia_${dataset.fileName}.csv`)}
            className="inline-flex items-center gap-1.5 px-3 py-2 border border-slate-200 bg-white hover:bg-slate-50 text-xs font-medium text-slate-700 rounded-lg transition-colors whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5" />
            Base Limpia (CSV)
          </button>
          {simRows.length > 0 && (
            <button
              type="button"
              onClick={() =>
                exportRowsToCsv(
                  simRows.map((r) => ({
                    Periodo: r.periodIndex,
                    Origen: r.sourceType,
                    ...r.values,
                    Y_Estimado: Number(r.predictedY.toFixed(2)),
                  })),
                  `Datos_Simulados_${dataset.fileName}.csv`
                )
              }
              className="inline-flex items-center gap-1.5 px-3 py-2 border border-slate-200 bg-white hover:bg-slate-50 text-xs font-medium text-slate-700 rounded-lg transition-colors whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              Simulados (CSV)
            </button>
          )}
          <button
            type="button"
            onClick={() => exportScenariosToCsv(scenarios, selectedModel.targetColumn, `Escenarios_${dataset.fileName}.csv`)}
            className="inline-flex items-center gap-1.5 px-3 py-2 border border-slate-200 bg-white hover:bg-slate-50 text-xs font-medium text-slate-700 rounded-lg transition-colors whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5" />
            Escenarios (CSV)
          </button>
          <button
            type="button"
            onClick={() =>
              exportFullExcelWorkbook({
                datasetName: dataset.fileName,
                cleaned,
                models,
                selectedModel,
                selection,
                simPlans,
                simRows,
                scenarios,
                recommendations,
              })
            }
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap"
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            Libro Completo (.xlsx)
          </button>
          <button
            type="button"
            onClick={() =>
              exportExecutivePdfReport({
                datasetName: dataset.fileName,
                businessContext,
                cleaned,
                selectedModel,
                selection,
                scenarios,
                recommendations,
                narrative,
              })
            }
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-[#0B1F3A] hover:bg-slate-800 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap"
          >
            <FileText className="w-3.5 h-3.5" />
            Informe Ejecutivo (.pdf)
          </button>
        </div>
      </div>

      {/* Executive KPI Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-xs text-slate-500">Completitud Post-Limpieza</p>
          <p className="text-2xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
            {cleaned.completenessAfter.toFixed(1)}%
          </p>
          <p className="text-[11px] text-slate-500 mt-1">
            {cleaned.allCleanRows.length} reg. · Split {cleaned.splitConfig.trainPct}/{cleaned.splitConfig.valPct}/{cleaned.splitConfig.testPct}
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-xs text-slate-500">CR² Modelo Seleccionado</p>
          <p className="text-2xl font-bold text-teal-700 font-mono tabular-nums mt-1">
            {(selectedModel.adjR2Train * 100).toFixed(1)}%
          </p>
          <p className="text-[11px] text-slate-500 mt-1 truncate">{selectedModel.name}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-xs text-slate-500">RMSE Fuera de Muestra</p>
          <p className="text-2xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
            {formatNum(selectedModel.rmseVal, 2)}
          </p>
          <p className="text-[11px] text-slate-500 mt-1">CV: {formatNum(selectedModel.cvRmseMean, 2)}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-xs text-slate-500">Escenario Base ({selectedModel.targetColumn})</p>
          <p className="text-2xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
            {baseScenario ? formatNum(baseScenario.predictedY, 1) : "—"}
          </p>
          <p className="text-[11px] text-slate-500 mt-1">{targetUnit}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-xs text-slate-500">Potencial Optimista</p>
          <p className="text-2xl font-bold text-teal-700 font-mono tabular-nums mt-1">
            {optScenario ? `${optScenario.pctVsBase >= 0 ? "+" : ""}${optScenario.pctVsBase.toFixed(1)}%` : "—"}
          </p>
          <p className="text-[11px] text-slate-500 mt-1">
            {optScenario ? `Ŷ = ${formatNum(optScenario.predictedY, 1)}` : ""}
          </p>
        </div>
      </div>

      {/* Narrative Generator + Scenario Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="text-sm font-semibold text-[#0B1F3A]">Síntesis Ejecutiva para Dirección</h3>
            <button
              type="button"
              onClick={runNarrative}
              disabled={genBusy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg transition-colors disabled:opacity-50"
            >
              <Sparkles className="w-3.5 h-3.5" />
              {genBusy ? "Redactando..." : narrative ? "Actualizar Síntesis" : "Redactar Informe IA"}
            </button>
          </div>

          {genError && <p className="text-xs text-red-600">{genError}</p>}

          {narrative ? (
            <div className="space-y-3 text-xs leading-relaxed text-slate-700">
              <p className="whitespace-pre-line">{narrative.executiveSummary}</p>
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <p className="font-semibold text-[#0B1F3A] mb-1">Notas Metodológicas</p>
                <p>{narrative.methodologicalNotes}</p>
              </div>
              <div>
                <p className="font-semibold text-[#0B1F3A] mb-1">Plan de Acción Sugerido</p>
                <ul className="list-disc pl-4 space-y-1">
                  {narrative.actionPlan.map((act, i) => (
                    <li key={i}>{act}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <div className="space-y-2 text-xs text-slate-600">
              <p>
                <strong>Ecuación Activa:</strong> <span className="font-mono">{selectedModel.equation}</span>
              </p>
              <p>
                <strong>Validación F:</strong>{" "}
                {selectedModel.fStat !== null
                  ? `F=${formatNum(selectedModel.fStat, 2)} (p=${formatPValue(selectedModel.fPValue)})`
                  : "Modelo regularizado"}
              </p>
              <p>
                <strong>Criterio de Selección:</strong> {selection.rationale}
              </p>
              <p className="text-slate-400 pt-2">
                Haz clic en "Redactar Informe IA" para generar la narrativa ejecutiva completa antes de descargar el PDF.
              </p>
            </div>
          )}
        </div>

        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <h3 className="text-sm font-semibold text-[#0B1F3A]">Escenarios Comparados ({selectedModel.targetColumn})</h3>
          <ScenarioComparisonChart scenarios={scenarios} targetUnit={targetUnit} />
        </div>
      </div>

      {/* Summary of Auxiliary Simulation Methods */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-[#0B1F3A]">
            Resumen de Modelos Auxiliares por Variable y Diagnósticos
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="py-2 px-3 font-medium">Predictor</th>
                <th className="py-2 px-3 font-medium">Estrategia Simulación</th>
                <th className="py-2 px-3 font-medium">Ajuste / Rango</th>
                <th className="py-2 px-3 font-medium text-right">Coef. β (Modelo)</th>
                <th className="py-2 px-3 font-medium text-right">p-valor</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {simPlans.map((p) => {
                const c = selectedModel.coefficients.find((cf) => cf.sourceColumn === p.column);
                return (
                  <tr key={p.column}>
                    <td className="py-2 px-3 font-semibold text-[#0B1F3A]">{p.column}</td>
                    <td className="py-2 px-3 text-slate-700">{p.selectedStrategy}</td>
                    <td className="py-2 px-3 font-mono text-slate-600">{p.auxFitQuality}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {c ? formatNum(c.estimate, 4) : "Dummy"}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {c ? formatPValue(c.pValue) : "Múltiple"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
