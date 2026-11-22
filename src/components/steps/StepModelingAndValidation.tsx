import React, { useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Play,
  SlidersHorizontal,
  TrendingUp,
} from "lucide-react";
import {
  CleanedDatasetResult,
  DatasetState,
  GeminiContextAnalysis,
  ModelSelectionResult,
  SelectionFocus,
  TrainedModel,
} from "../../types/analytics";
import { formatNum, formatPValue } from "../../engine/mathStats";
import { CoefficientForestPlot, QQPlotChart, ResidualsScatterChart } from "../Charts";

export const StepVariableSelection: React.FC<{
  dataset: DatasetState | null;
  cleaned: CleanedDatasetResult | null;
  targetColumn: string | null;
  selectedPredictors: string[];
  includeRegularization: boolean;
  geminiAnalysis: GeminiContextAnalysis | null;
  guidedMode: boolean;
  onSelectTarget: (col: string) => void;
  onTogglePredictor: (col: string) => void;
  onToggleRegularization: (val: boolean) => void;
  onTrainModels: () => void;
}> = ({
  dataset,
  cleaned,
  targetColumn,
  selectedPredictors,
  includeRegularization,
  geminiAnalysis,
  guidedMode,
  onSelectTarget,
  onTogglePredictor,
  onToggleRegularization,
  onTrainModels,
}) => {
  if (!dataset || !cleaned) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Ejecuta primero la limpieza y split en el paso 03.</p>
      </div>
    );
  }

  const targetProfile = dataset.columns.find((c) => c.name === targetColumn);
  const isTargetValidForLinear =
    targetProfile &&
    (targetProfile.type === "numeric_continuous" || targetProfile.type === "numeric_discrete") &&
    !targetProfile.isConstant &&
    targetProfile.uniqueCount >= 5;

  const getPredictorTier = (colName: string): { tier: "Recomendada" | "Opcional" | "No recomendada"; reason: string } => {
    const col = dataset.columns.find((c) => c.name === colName);
    if (!col) return { tier: "Opcional", reason: "Disponible para evaluación." };
    if (col.isConstant) return { tier: "No recomendada", reason: "Variable constante sin varianza explicativa." };
    if (col.type === "textual") return { tier: "No recomendada", reason: "Texto libre de alta cardinalidad o identificador." };
    if (col.type === "temporal") return { tier: "No recomendada", reason: "Columna de fecha cruda (usar para orden temporal o extraer componentes)." };
    if (col.businessRole === "identifier_exclude") {
      return { tier: "No recomendada", reason: "Marcada como identificador o excluida en el diccionario de datos." };
    }
    const aiMatch = geminiAnalysis?.predictorAssessments.find(
      (p) => p.column.toLowerCase() === colName.toLowerCase()
    );
    if (aiMatch) {
      const normalizedTier =
        aiMatch.tier.toLowerCase().includes("no")
          ? "No recomendada"
          : aiMatch.tier.toLowerCase().includes("opc")
          ? "Opcional"
          : "Recomendada";
      return { tier: normalizedTier, reason: aiMatch.reason };
    }
    if (col.businessRole === "controllable") {
      return { tier: "Recomendada", reason: "Palanca accionable por el negocio con variabilidad histórica." };
    }
    if (col.type === "categorical" && col.uniqueCount > 10) {
      return { tier: "Opcional", reason: `Alta cardinalidad (${col.uniqueCount} categorías) puede consumir grados de libertad.` };
    }
    return { tier: "Recomendada", reason: "Plausibilidad económica y tipo de dato compatible con regresión OLS." };
  };

  const candidates = dataset.columns.filter((c) => c.name !== targetColumn);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">05. Selección Inteligente de Variables</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Variable Dependiente (Y) y Predictores Candidatos (X)
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Propuesta fundamentada en tu pregunta empresarial, viabilidad econométrica, ausencia de fugas y
              disponibilidad temporal — nunca basada únicamente en correlaciones altas.
            </p>
          )}
        </div>
        {isTargetValidForLinear && selectedPredictors.length >= 1 && (
          <button
            onClick={onTrainModels}
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-teal-600 hover:bg-teal-700 text-white text-sm font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Play className="w-4 h-4" />
            Entrenar y Comparar Modelos
          </button>
        )}
      </div>

      {/* Target Y Selection */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-[#0B1F3A]">1. Selección y Validación de Variable Dependiente (Y)</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {dataset.columns
            .filter((c) => c.type !== "temporal" && c.type !== "textual")
            .map((col) => {
              const isSelected = col.name === targetColumn;
              const apt =
                (col.type === "numeric_continuous" || col.type === "numeric_discrete") &&
                !col.isConstant &&
                col.uniqueCount >= 5;
              const aiTarget = geminiAnalysis?.suggestedTargets.find(
                (t) => t.column.toLowerCase() === col.name.toLowerCase()
              );
              return (
                <button
                  key={col.name}
                  type="button"
                  onClick={() => onSelectTarget(col.name)}
                  className={`text-left p-4 rounded-lg border transition-colors ${
                    isSelected
                      ? "border-teal-600 bg-teal-50/40"
                      : "border-slate-200 hover:border-slate-300 bg-white"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-[#0B1F3A]">{col.name}</span>
                    <span className={`text-[11px] font-medium ${apt ? "text-teal-700" : "text-red-700"}`}>
                      {apt ? "Apta OLS" : "No apta OLS"}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    {col.type} · {col.uniqueCount} valores únicos · {col.unit}
                  </p>
                  {aiTarget && <p className="text-xs text-slate-700 mt-1.5">{aiTarget.reason}</p>}
                </button>
              );
            })}
        </div>

        {!isTargetValidForLinear && targetProfile && (
          <div className="p-4 rounded-xl bg-red-50 border border-red-200 flex items-start gap-3 text-xs text-red-900">
            <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-semibold text-sm">
                "{targetProfile.name}" no es adecuada para Regresión Lineal Clásica (OLS)
              </p>
              <p>
                Tipo detectado: <strong>{targetProfile.type}</strong> ({targetProfile.uniqueCount} valores únicos). La
                regresión lineal requiere una variable respuesta continua o cuantitativa con suficiente rango. Si
                deseas predecir una variable binaria (0/1, Sí/No) o categórica, la técnica correcta es{" "}
                <strong>Regresión Logística</strong> o modelos de clasificación supervisada. Selecciona una variable
                numérica continua arriba para continuar con regresión lineal.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Predictor X Selection */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[#0B1F3A]">
              2. Selección de Variables Independientes (X) — {selectedPredictors.length} activas
            </h3>
            <p className="text-xs text-slate-500">
              Las variables categóricas seleccionadas se codificarán automáticamente como variables dummy usando su categoría modal como referencia.
            </p>
          </div>
          <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={includeRegularization}
              onChange={(e) => onToggleRegularization(e.target.checked)}
              className="rounded border-slate-300 text-teal-600"
            />
            Incluir Ridge (L2) y Lasso (L1) como complemento
          </label>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-medium">Incluir</th>
                <th className="py-2.5 px-3 font-medium">Variable (X)</th>
                <th className="py-2.5 px-3 font-medium">Tipo</th>
                <th className="py-2.5 px-3 font-medium">Clasificación</th>
                <th className="py-2.5 px-3 font-medium">Rol</th>
                <th className="py-2.5 px-3 font-medium">Criterio de Selección</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {candidates.map((col) => {
                const { tier, reason } = getPredictorTier(col.name);
                const checked = selectedPredictors.includes(col.name);
                const disabled = col.isConstant || col.type === "textual" || col.type === "temporal";
                return (
                  <tr key={col.name} className="hover:bg-slate-50/50">
                    <td className="py-2.5 px-3">
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={disabled}
                        onChange={() => onTogglePredictor(col.name)}
                        className="rounded border-slate-300 text-teal-600 disabled:opacity-40"
                      />
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-[#0B1F3A]">{col.name}</td>
                    <td className="py-2.5 px-3 text-slate-600">{col.type}</td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`font-medium ${
                          tier === "Recomendada"
                            ? "text-teal-700"
                            : tier === "Opcional"
                            ? "text-amber-700"
                            : "text-red-700"
                        }`}
                      >
                        {tier}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-600">
                      {col.businessRole === "controllable" ? "Modificable" : "Externa / Fija"}
                    </td>
                    <td className="py-2.5 px-3 text-slate-600 max-w-md">{reason}</td>
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

export const StepModeling: React.FC<{
  models: TrainedModel[];
  activeModelId: string | null;
  guidedMode: boolean;
  onSelectActiveModel: (id: string) => void;
  onNext: () => void;
}> = ({ models, activeModelId, guidedMode, onSelectActiveModel, onNext }) => {
  const model = models.find((m) => m.id === activeModelId) ?? models[0];

  if (!model) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Entrena primero los modelos en el paso 05.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">06. Construcción de Regresiones Lineales</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Estimación OLS, Regularización y Diagnóstico de Residuos
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Inspecciona la ecuación ajustada, significancia individual (t y p-valores), significancia global (F) y
              supuestos sobre residuos (independencia, homocedasticidad y normalidad).
            </p>
          )}
        </div>
        <button
          onClick={onNext}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
        >
          Ir al Comparador Multicriterio
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Model Selector Tabs */}
      <div className="flex flex-wrap items-center gap-1.5 p-1.5 bg-slate-100 rounded-xl">
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => onSelectActiveModel(m.id)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors whitespace-nowrap ${
              m.id === model.id ? "bg-white text-[#0B1F3A] shadow-xs" : "text-slate-600 hover:text-slate-900"
            }`}
          >
            {m.name}
          </button>
        ))}
      </div>

      {/* Equation Banner */}
      <div className="bg-[#0B1F3A] text-white rounded-xl p-5 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs text-teal-300 font-medium">
            {model.method === "OLS_SIMPLE"
              ? "Regresión Lineal Simple (OLS)"
              : model.method === "OLS_MULTIPLE"
              ? "Regresión Lineal Múltiple (OLS)"
              : `Regularización ${model.method} (α=${model.regularizationAlpha})`}
          </span>
          <span className="text-xs font-mono text-slate-300 tabular-nums">
            Train n={model.nTrain} · Val n={model.nVal} · Test n={model.nTest}
          </span>
        </div>
        <p className="font-mono text-xs md:text-sm leading-relaxed text-slate-100 break-words">
          {model.equation}
        </p>
      </div>

      {!model.inferenceValid && model.inferenceWarning && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-900">
          <strong>Nota inferencial:</strong> {model.inferenceWarning}
        </div>
      )}

      {/* Metrics Strip */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-white border border-slate-200 rounded-lg p-3.5">
          <p className="text-xs text-slate-500">R² (Train)</p>
          <p className="text-xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
            {(model.r2Train * 100).toFixed(1)}%
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-3.5">
          <p className="text-xs text-slate-500">CR² (R² Ajustado)</p>
          <p className="text-xl font-bold text-teal-700 font-mono tabular-nums mt-1">
            {(model.adjR2Train * 100).toFixed(1)}%
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-3.5">
          <p className="text-xs text-slate-500">RMSE Validación</p>
          <p className="text-xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
            {formatNum(model.rmseVal, 2)}
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-3.5">
          <p className="text-xs text-slate-500">MAE Validación</p>
          <p className="text-xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
            {formatNum(model.maeVal, 2)}
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-3.5">
          <p className="text-xs text-slate-500">Estadístico F (p-val)</p>
          <p className="text-base font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
            {model.fStat !== null ? `${formatNum(model.fStat, 2)} (${formatPValue(model.fPValue)})` : "No aplica"}
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-3.5">
          <p className="text-xs text-slate-500">CV RMSE ({model.cvMethod.split(" ")[0]})</p>
          <p className="text-xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
            {formatNum(model.cvRmseMean, 2)}
          </p>
        </div>
      </div>

      {/* Coefficients Table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-[#0B1F3A]">Tabla de Coeficientes e Inferencia</h3>
          <span className="text-xs text-slate-500">Error estándar residual (s): {formatNum(model.residualStdError, 3)}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-medium">Término</th>
                <th className="py-2.5 px-3 font-medium text-right">Coeficiente (β)</th>
                <th className="py-2.5 px-3 font-medium text-right">Error Est.</th>
                <th className="py-2.5 px-3 font-medium text-right">Estadístico t</th>
                <th className="py-2.5 px-3 font-medium text-right">p-valor</th>
                <th className="py-2.5 px-3 font-medium text-right">IC 95%</th>
                <th className="py-2.5 px-3 font-medium text-right">VIF</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {model.coefficients.map((c) => {
                const sig = c.pValue !== null && c.pValue < 0.05;
                return (
                  <tr key={c.term} className="hover:bg-slate-50/50">
                    <td className="py-2.5 px-3 font-semibold text-[#0B1F3A]">
                      {c.term}
                      {c.referenceCategory && (
                        <span className="block text-[11px] font-normal text-slate-500">
                          Ref: {c.referenceCategory}
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums font-semibold">
                      {formatNum(c.estimate, 4)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                      {c.stdError !== null ? formatNum(c.stdError, 4) : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                      {c.tStat !== null ? formatNum(c.tStat, 3) : "—"}
                    </td>
                    <td
                      className={`py-2.5 px-3 text-right font-mono tabular-nums ${
                        c.pValue === null
                          ? "text-slate-400"
                          : sig
                          ? "text-teal-700 font-semibold"
                          : "text-amber-700"
                      }`}
                    >
                      {formatPValue(c.pValue)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                      {c.ciLow !== null && c.ciHigh !== null
                        ? `[${formatNum(c.ciLow, 2)}, ${formatNum(c.ciHigh, 2)}]`
                        : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                      {c.vif !== null ? formatNum(c.vif, 2) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Residual Diagnostics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <h3 className="text-sm font-semibold text-[#0B1F3A]">Residuos vs Valores Ajustados</h3>
          <ResidualsScatterChart data={model.diagnostics.residualsVsFitted} />
          <div className="text-xs text-slate-600 space-y-1 pt-2 border-t border-slate-100">
            <p>
              <strong>Breusch-Pagan (Homocedasticidad):</strong> LM = {formatNum(model.diagnostics.breuschPaganStat, 2)} · p ={" "}
              {formatPValue(model.diagnostics.breuschPaganPValue)} — {model.diagnostics.breuschPaganNote}
            </p>
            <p>
              <strong>Durbin-Watson (Independencia):</strong> DW = {formatNum(model.diagnostics.durbinWatson, 3)} —{" "}
              {model.diagnostics.durbinWatsonNote}
            </p>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <h3 className="text-sm font-semibold text-[#0B1F3A]">Gráfico Q-Q de Normalidad de Residuos</h3>
          <QQPlotChart data={model.diagnostics.qqPoints} />
          <div className="text-xs text-slate-600 pt-2 border-t border-slate-100">
            <p>
              <strong>Jarque-Bera (Normalidad):</strong> JB = {formatNum(model.diagnostics.jarqueBeraStat, 2)} · p ={" "}
              {formatPValue(model.diagnostics.jarqueBeraPValue)} — {model.diagnostics.jarqueBeraNote}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export const StepValidationComparator: React.FC<{
  models: TrainedModel[];
  selection: ModelSelectionResult | null;
  focus: SelectionFocus;
  activeModelId: string | null;
  guidedMode: boolean;
  onChangeFocus: (f: SelectionFocus) => void;
  onSelectModel: (id: string) => void;
  onNext: () => void;
}> = ({ models, selection, focus, activeModelId, guidedMode, onChangeFocus, onSelectModel, onNext }) => {
  if (!selection || models.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Entrena primero los modelos candidatos en el paso 05.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">07. Validación y Selección del Mejor Modelo</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Comparador Multicriterio: CR², RMSE Fuera de Muestra y Pruebas t
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Evaluación transparente sin puntajes artificiales: prioriza RMSE fuera de muestra para pronóstico o
              CR² e inferencia t válida para explicación causal/económica.
            </p>
          )}
        </div>
        <button
          onClick={onNext}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
        >
          Interpretar Coeficientes
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Configurable Criteria Selector */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-xs font-semibold text-[#0B1F3A]">
          <SlidersHorizontal className="w-4 h-4 text-teal-600" />
          <span>Objetivo de Selección Multicriterio:</span>
        </div>
        <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 rounded-lg">
          {(
            [
              { id: "balanced", label: "Equilibrado (CR² + RMSE + t)" },
              { id: "predictive", label: "Predictivo (Prioriza RMSE fuera de muestra)" },
              { id: "explanatory", label: "Explicativo (Prioriza CR² y Prueba t)" },
            ] as const
          ).map((opt) => (
            <button
              key={opt.id}
              onClick={() => onChangeFocus(opt.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                focus === opt.id ? "bg-white text-[#0B1F3A] shadow-xs" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Recommendation Summary */}
      <div
        className={`border rounded-xl p-6 space-y-4 ${
          selection.noReliableModel
            ? "bg-red-50 border-red-200 text-red-950"
            : "bg-white border-slate-200 text-slate-800"
        }`}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-start gap-2.5">
            {selection.noReliableModel ? (
              <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
            ) : (
              <CheckCircle2 className="w-5 h-5 text-teal-600 shrink-0 mt-0.5" />
            )}
            <div>
              <h3 className="text-base font-bold text-[#0B1F3A]">{selection.headline}</h3>
              <p className="text-sm text-slate-600 mt-1">{selection.rationale}</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-2 border-t border-slate-100 text-xs">
          <div>
            <p className="font-semibold text-[#0B1F3A] mb-1">Ventajas</p>
            <ul className="list-disc pl-4 space-y-1 text-slate-600">
              {selection.advantages.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          </div>
          <div>
            <p className="font-semibold text-[#0B1F3A] mb-1">Limitaciones</p>
            <ul className="list-disc pl-4 space-y-1 text-slate-600">
              {selection.limitations.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          </div>
          <div>
            <p className="font-semibold text-[#0B1F3A] mb-1">Variables que Contribuyen</p>
            <p className="text-slate-600">{selection.contributingVariables.join(", ") || "Ninguna significativa"}</p>
          </div>
          <div>
            <p className="font-semibold text-[#0B1F3A] mb-1">Supuestos y Diagnósticos</p>
            <ul className="list-disc pl-4 space-y-1 text-slate-600">
              {selection.attentionAssumptions.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-[#0B1F3A]">
            Tabla Comparativa de Modelos (Haz clic para activar modelo en Simulación)
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-medium">Modelo</th>
                <th className="py-2.5 px-3 font-medium text-right">CR² (Ajustado)</th>
                <th className="py-2.5 px-3 font-medium text-right">RMSE Val.</th>
                <th className="py-2.5 px-3 font-medium text-right">RMSE CV</th>
                <th className="py-2.5 px-3 font-medium text-right">MAE Val.</th>
                <th className="py-2.5 px-3 font-medium">Prueba t / p-val</th>
                <th className="py-2.5 px-3 font-medium text-right">Prueba F (p)</th>
                <th className="py-2.5 px-3 font-medium text-right">VIF Máx</th>
                <th className="py-2.5 px-3 font-medium">Residuos</th>
                <th className="py-2.5 px-3 font-medium">Estado</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {selection.comparisonRows.map((row) => {
                const isActive = row.modelId === activeModelId;
                const isRecommended = row.modelId === selection.recommendedModelId;
                return (
                  <tr
                    key={row.modelId}
                    onClick={() => onSelectModel(row.modelId)}
                    className={`cursor-pointer transition-colors ${
                      isActive ? "bg-teal-50/70" : "hover:bg-slate-50/70"
                    }`}
                  >
                    <td className="py-2.5 px-3 font-semibold text-[#0B1F3A]">
                      {row.name}
                      {isRecommended && (
                        <span className="block text-[11px] font-medium text-teal-700">Recomendado</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums font-semibold">
                      {(row.adjR2 * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums font-semibold">
                      {formatNum(row.rmseVal, 2)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">{formatNum(row.cvRmse, 2)}</td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">{formatNum(row.maeVal, 2)}</td>
                    <td className="py-2.5 px-3 text-slate-700">{row.significantCoefRatio}</td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                      {formatPValue(row.fPValue)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">{formatNum(row.maxVif, 2)}</td>
                    <td className="py-2.5 px-3 text-slate-700">{row.residualHealth}</td>
                    <td className="py-2.5 px-3">
                      <span className={isActive ? "font-semibold text-teal-700" : "text-slate-500"}>
                        {isActive ? "Activo" : "Seleccionar"}
                      </span>
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

export const StepInterpretation: React.FC<{
  dataset: DatasetState | null;
  model: TrainedModel | null;
  guidedMode: boolean;
  onNext: () => void;
}> = ({ dataset, model, guidedMode, onNext }) => {
  const [filterSigOnly, setFilterSigOnly] = useState(false);

  if (!model || !dataset) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Selecciona o entrena un modelo en los pasos 05–07.</p>
      </div>
    );
  }

  const targetCol = dataset.columns.find((c) => c.name === model.targetColumn);
  const targetUnit = targetCol?.unit ?? "unidades";

  const coefs = model.coefficients.filter((c) => {
    if (c.isIntercept) return false;
    if (filterSigOnly && model.inferenceValid) return c.pValue !== null && c.pValue < 0.05;
    return true;
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">08. Interpretación Empresarial de Coeficientes</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Lectura Económica y Estadística · {model.name}
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Cada tarjeta traduce la magnitud del coeficiente, su intervalo de confianza y su significancia a lenguaje
              de negocio manteniendo constantes los demás predictores (ceteris paribus), sin inferir causalidad automática.
            </p>
          )}
        </div>
        <button
          onClick={onNext}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
        >
          Ir al Motor de Simulación
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Forest Plot */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-[#0B1F3A]">
              Gráfico de Coeficientes con Intervalos de Confianza (95%)
            </h3>
            <p className="text-xs text-slate-500">
              Si el intervalo de confianza cruza la línea cero, el efecto individual no es estadísticamente distinguible de cero al 5%.
            </p>
          </div>
          {model.inferenceValid && (
            <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={filterSigOnly}
                onChange={(e) => setFilterSigOnly(e.target.checked)}
                className="rounded border-slate-300 text-teal-600"
              />
              Mostrar solo significativos (p &lt; 0.05)
            </label>
          )}
        </div>
        <CoefficientForestPlot coefficients={model.coefficients} />
      </div>

      {/* Interactive Coefficient Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {coefs.map((c) => {
          const srcCol = dataset.columns.find((col) => col.name === c.sourceColumn);
          const xUnit = srcCol?.unit ?? "unidad";
          const dirWord = c.estimate >= 0 ? "incremento" : "reducción";
          const isSig = c.pValue !== null && c.pValue < 0.05;

          const statInterpretation = model.inferenceValid
            ? isSig
              ? `Evidencia estadística fuerte (t=${formatNum(c.tStat, 2)}, p=${formatPValue(c.pValue)}): se rechaza H₀: β=0 al 5%.`
              : `Sin evidencia suficiente al 5% (t=${formatNum(c.tStat, 2)}, p=${formatPValue(c.pValue)}): no se descarta efecto nulo.`
            : "Coeficiente estimado bajo penalización regularizada (sin p-valor clásico de OLS).";

          const bizInterpretation = c.isDummy
            ? `Pertenecer a la categoría "${c.term.split("[")[1]?.replace("]", "")}" frente a la categoría de referencia "${c.referenceCategory}" se asocia con un ${dirWord} promedio estimado de ${formatNum(Math.abs(c.estimate), 2)} ${targetUnit} en ${model.targetColumn}, manteniendo constantes las demás variables.`
            : `Un incremento de 1 unidad (${xUnit}) en ${c.sourceColumn} se asocia con una variación estimada de ${c.estimate >= 0 ? "+" : ""}${formatNum(c.estimate, 2)} ${targetUnit} en ${model.targetColumn}, manteniendo constantes los demás predictores (asociación condicional, no causalidad demostrada).`;

          return (
            <div key={c.term} className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
              <div className="flex items-start justify-between gap-2 border-b border-slate-100 pb-3">
                <div>
                  <p className="text-xs text-slate-500">
                    {c.isDummy ? `Variable Dummy · Ref: ${c.referenceCategory}` : `Unidad X: ${xUnit}`}
                  </p>
                  <h4 className="text-base font-bold text-[#0B1F3A] mt-0.5 flex items-center gap-1.5">
                    <TrendingUp className={`w-4 h-4 ${c.estimate >= 0 ? "text-teal-600" : "text-red-600"}`} />
                    {c.term}
                  </h4>
                </div>
                <div className="text-right font-mono tabular-nums">
                  <p className={`text-lg font-bold ${c.estimate >= 0 ? "text-teal-700" : "text-red-700"}`}>
                    {c.estimate >= 0 ? "+" : ""}
                    {formatNum(c.estimate, 4)}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    {c.pValue !== null ? `p = ${formatPValue(c.pValue)}` : "Regularizado"}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono tabular-nums text-slate-600">
                <div>
                  IC 95%:{" "}
                  {c.ciLow !== null && c.ciHigh !== null
                    ? `[${formatNum(c.ciLow, 2)}, ${formatNum(c.ciHigh, 2)}]`
                    : "No aplica"}
                </div>
                <div className="text-right">
                  VIF: {c.vif !== null ? formatNum(c.vif, 2) : "—"}
                </div>
              </div>

              <div className="space-y-2 text-xs leading-relaxed">
                <p className="text-slate-600">
                  <strong className="text-slate-900">Lectura estadística:</strong> {statInterpretation}
                </p>
                <p className="text-slate-700 bg-slate-50 border border-slate-200 rounded-lg p-3">
                  <strong className="text-[#0B1F3A]">Lectura empresarial:</strong> {bizInterpretation}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
