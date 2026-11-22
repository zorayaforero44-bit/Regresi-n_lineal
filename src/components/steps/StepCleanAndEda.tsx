import React, { useMemo, useState } from "react";
import { ArrowRight, CheckCircle2, ShieldAlert, Play, AlertCircle } from "lucide-react";
import {
  CleanedDatasetResult,
  CleaningColumnRule,
  CorrelationMethod,
  DatasetState,
  ImputationStrategy,
  SplitConfig,
} from "../../types/analytics";
import {
  buildHistogramBins,
  buildTimeSeriesPoints,
  computeCategoricalAssociations,
  computeCorrelationMatrix,
  computeVIF,
  getNumericValuesPair,
} from "../../engine/edaEngine";
import { formatNum, iqrBounds } from "../../engine/mathStats";
import {
  BoxplotSvg,
  CorrelationHeatmap,
  HistogramChart,
  ScatterTrendChart,
  TimeTrendChart,
} from "../Charts";

const STRATEGY_OPTIONS: Array<{ value: ImputationStrategy; label: string; group: string }> = [
  { value: "none", label: "Sin imputación", group: "General" },
  { value: "mean", label: "Media (Train)", group: "Numérica" },
  { value: "median", label: "Mediana (Train)", group: "Numérica" },
  { value: "group_median", label: "Mediana por grupo (Train)", group: "Numérica" },
  { value: "knn", label: "KNN Multivariado (Train)", group: "Numérica" },
  { value: "mode", label: "Moda (Train)", group: "Categórica" },
  { value: "missing_category", label: "Categoría explícita 'Sin Dato'", group: "Categórica" },
  { value: "group_mode", label: "Moda por grupo (Train)", group: "Categórica" },
  { value: "forward_fill", label: "Arrastre temporal (solo pasado)", group: "Temporal" },
  { value: "drop_rows", label: "Excluir nulos de Y (obligatorio)", group: "Objetivo Y" },
];

export const StepCleaningAndSplit: React.FC<{
  dataset: DatasetState | null;
  targetColumn: string | null;
  rules: CleaningColumnRule[];
  splitConfig: SplitConfig;
  deduplicate: boolean;
  cleaned: CleanedDatasetResult | null;
  guidedMode: boolean;
  onChangeTargetPreview: (col: string) => void;
  onUpdateRule: (col: string, patch: Partial<CleaningColumnRule>) => void;
  onChangeSplit: (patch: Partial<SplitConfig>) => void;
  onToggleDeduplicate: (val: boolean) => void;
  onExecuteCleaning: () => void;
  onNext: () => void;
}> = ({
  dataset,
  targetColumn,
  rules,
  splitConfig,
  deduplicate,
  cleaned,
  guidedMode,
  onChangeTargetPreview,
  onUpdateRule,
  onChangeSplit,
  onToggleDeduplicate,
  onExecuteCleaning,
  onNext,
}) => {
  if (!dataset || dataset.rawRows.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Primero carga una base de datos en el paso 01.</p>
      </div>
    );
  }

  const dateCols = dataset.columns.filter((c) => c.type === "temporal");
  const catCols = dataset.columns.filter(
    (c) => c.type === "categorical" || c.type === "ordinal" || c.type === "binary"
  );
  const numericCols = dataset.columns.filter(
    (c) => c.type === "numeric_continuous" || c.type === "numeric_discrete"
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">03. Limpieza, Imputación y Split Anti-Fuga</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Preparación Estadística sin Fuga de Información
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Primero separamos los conjuntos de Entrenamiento, Validación y Prueba; luego ajustamos imputadores
              diferenciados únicamente sobre Entrenamiento. La variable objetivo (Y) nunca se imputa con valores predichos.
            </p>
          )}
        </div>
        {cleaned && (
          <button
            onClick={onNext}
            className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Ir a Exploración (EDA)
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Split Config Card */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-[#0B1F3A]">
              Configuración de Split Previo al Ajuste de Imputadores
            </h3>
            <p className="text-xs text-slate-500">
              Garantiza evaluación honesta fuera de muestra (Train / Validación / Test).
            </p>
          </div>
          <div className="flex items-center gap-3">
            <label className="inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={deduplicate}
                onChange={(e) => onToggleDeduplicate(e.target.checked)}
                className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
              />
              Eliminar duplicados exactos ({dataset.duplicateRowCount})
            </label>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Variable Objetivo Tentativa (Y)</label>
            <select
              value={targetColumn ?? ""}
              onChange={(e) => onChangeTargetPreview(e.target.value)}
              className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-xs bg-white"
            >
              {numericCols.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
              {dataset.columns
                .filter((c) => !numericCols.some((n) => n.name === c.name))
                .map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name} ({c.type})
                  </option>
                ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Estrategia de Partición</label>
            <select
              value={splitConfig.mode}
              onChange={(e) =>
                onChangeSplit({
                  mode: e.target.value as "random" | "temporal",
                  dateColumn: splitConfig.dateColumn ?? dateCols[0]?.name,
                })
              }
              className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-xs bg-white"
            >
              <option value="random">Aleatorio reproducible (Seed)</option>
              {dateCols.length > 0 && <option value="temporal">Cronológico temporal (pasado → futuro)</option>}
            </select>
          </div>

          {splitConfig.mode === "temporal" && dateCols.length > 0 ? (
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Columna de Fecha</label>
              <select
                value={splitConfig.dateColumn ?? dateCols[0].name}
                onChange={(e) => onChangeSplit({ dateColumn: e.target.value })}
                className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-xs bg-white"
              >
                {dateCols.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Semilla (Seed)</label>
              <input
                type="number"
                value={splitConfig.seed}
                onChange={(e) => onChangeSplit({ seed: Number(e.target.value) || 42 })}
                className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-xs font-mono"
              />
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">% Entrenamiento / Validación / Prueba</label>
            <div className="flex items-center gap-1.5">
              <input
                type="number"
                min={50}
                max={85}
                value={splitConfig.trainPct}
                onChange={(e) => {
                  const tr = Math.max(50, Math.min(85, Number(e.target.value) || 70));
                  const rem = 100 - tr;
                  const val = Math.floor(rem / 2);
                  onChangeSplit({ trainPct: tr, valPct: val, testPct: rem - val });
                }}
                className="w-16 border border-slate-200 rounded-lg px-2 py-2 text-xs font-mono text-center"
              />
              <span className="text-xs text-slate-400">/</span>
              <input
                type="number"
                min={5}
                max={30}
                value={splitConfig.valPct}
                onChange={(e) => {
                  const val = Math.max(5, Math.min(30, Number(e.target.value) || 15));
                  const test = Math.max(5, 100 - splitConfig.trainPct - val);
                  onChangeSplit({ valPct: val, testPct: test });
                }}
                className="w-16 border border-slate-200 rounded-lg px-2 py-2 text-xs font-mono text-center"
              />
              <span className="text-xs text-slate-400">/</span>
              <span className="text-xs font-mono font-semibold text-slate-700 w-10 text-center">
                {splitConfig.testPct}%
              </span>
            </div>
          </div>

          <div className="flex items-end">
            <button
              type="button"
              onClick={onExecuteCleaning}
              className="w-full inline-flex items-center justify-center gap-2 py-2 px-4 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg transition-colors"
            >
              <Play className="w-3.5 h-3.5" />
              Ejecutar Pipeline Limpio
            </button>
          </div>
        </div>
      </div>

      {/* Imputation Strategy Matrix */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-[#0B1F3A]">
            Estrategia de Limpieza e Imputación Diferenciada por Variable
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            El sistema recomienda el método según asimetría, atípicos y tipo de dato; puedes ajustarlo libremente.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-medium">Variable</th>
                <th className="py-2.5 px-3 font-medium">Nulos / Atípicos</th>
                <th className="py-2.5 px-3 font-medium">Estrategia Seleccionada</th>
                <th className="py-2.5 px-3 font-medium">Grupo (si aplica)</th>
                <th className="py-2.5 px-3 font-medium">Tratamiento de Extremos</th>
                <th className="py-2.5 px-3 font-medium">Justificación Estadística</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rules.map((rule) => {
                const col = dataset.columns.find((c) => c.name === rule.column);
                const isTarget = rule.column === targetColumn;
                const needsGroup =
                  rule.selectedStrategy === "group_median" || rule.selectedStrategy === "group_mode";
                return (
                  <tr key={rule.column} className="hover:bg-slate-50/50">
                    <td className="py-2.5 px-3 font-semibold text-[#0B1F3A] whitespace-nowrap">
                      {rule.column}
                      {isTarget && (
                        <span className="block text-[11px] font-medium text-teal-700">Objetivo (Y)</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-slate-600 whitespace-nowrap">
                      {col?.missingCount ?? 0} nulos · {col?.outlierCount ?? 0} IQR
                    </td>
                    <td className="py-2 px-3">
                      <select
                        value={rule.selectedStrategy}
                        disabled={isTarget}
                        onChange={(e) =>
                          onUpdateRule(rule.column, {
                            selectedStrategy: e.target.value as ImputationStrategy,
                          })
                        }
                        className="border border-slate-200 rounded-md px-2 py-1 text-xs bg-white disabled:bg-slate-100"
                      >
                        {STRATEGY_OPTIONS.filter((opt) => {
                          if (isTarget) return opt.value === "drop_rows";
                          if (col?.type === "temporal") return opt.group === "General" || opt.group === "Temporal";
                          if (col?.type === "numeric_continuous" || col?.type === "numeric_discrete") {
                            return opt.group === "General" || opt.group === "Numérica";
                          }
                          return opt.group === "General" || opt.group === "Categórica";
                        }).map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="py-2 px-3">
                      {needsGroup ? (
                        <select
                          value={rule.groupByColumn ?? catCols[0]?.name ?? ""}
                          onChange={(e) => onUpdateRule(rule.column, { groupByColumn: e.target.value })}
                          className="border border-slate-200 rounded-md px-2 py-1 text-xs bg-white"
                        >
                          {catCols
                            .filter((c) => c.name !== rule.column)
                            .map((c) => (
                              <option key={c.name} value={c.name}>
                                {c.name}
                              </option>
                            ))}
                        </select>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      {col && (col.type === "numeric_continuous" || col.type === "numeric_discrete") && !isTarget ? (
                        <select
                          value={rule.outlierAction}
                          onChange={(e) =>
                            onUpdateRule(rule.column, {
                              outlierAction: e.target.value as "keep_flagged" | "winsorize_iqr",
                            })
                          }
                          className="border border-slate-200 rounded-md px-2 py-1 text-xs bg-white"
                        >
                          <option value="keep_flagged">Conservar y auditar contexto</option>
                          <option value="winsorize_iqr">Winsorizar a cotas IQR (Train)</option>
                        </select>
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-slate-600 max-w-md">{rule.reason}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Execution Summary */}
      {cleaned && (
        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2 text-[#0B1F3A] font-semibold text-sm">
              <CheckCircle2 className="w-4 h-4 text-teal-600" />
              <span>Resumen de Transformaciones y Split Ejecutado</span>
            </div>
            <span className="text-xs font-mono text-teal-700 tabular-nums">
              Completitud: {cleaned.completenessBefore.toFixed(1)}% → {cleaned.completenessAfter.toFixed(1)}%
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <p className="text-slate-500">Entrenamiento (Train)</p>
              <p className="text-lg font-bold text-[#0B1F3A] font-mono tabular-nums mt-0.5">
                {cleaned.trainRows.length}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <p className="text-slate-500">Validación (Val)</p>
              <p className="text-lg font-bold text-[#0B1F3A] font-mono tabular-nums mt-0.5">
                {cleaned.valRows.length}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <p className="text-slate-500">Prueba (Test)</p>
              <p className="text-lg font-bold text-[#0B1F3A] font-mono tabular-nums mt-0.5">
                {cleaned.testRows.length}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <p className="text-slate-500">Excluidos (Dupl. / Y nulo)</p>
              <p className="text-lg font-bold text-[#0B1F3A] font-mono tabular-nums mt-0.5">
                {cleaned.deduplicatedRows} / {cleaned.droppedTargetRows}
              </p>
            </div>
          </div>

          <p className="text-xs text-slate-600">{cleaned.learnedParametersNote}</p>

          {cleaned.changes.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                  <tr>
                    <th className="py-2 px-3 font-medium">Variable</th>
                    <th className="py-2 px-3 font-medium">Acción Aplicada</th>
                    <th className="py-2 px-3 font-medium text-right">Celdas/Filas</th>
                    <th className="py-2 px-3 font-medium">Detalle</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {cleaned.changes.map((ch, idx) => (
                    <tr key={idx}>
                      <td className="py-2 px-3 font-medium text-slate-900">{ch.column}</td>
                      <td className="py-2 px-3 text-teal-800">{ch.action}</td>
                      <td className="py-2 px-3 text-right font-mono tabular-nums">{ch.affectedCount}</td>
                      <td className="py-2 px-3 text-slate-600">{ch.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const StepEda: React.FC<{
  dataset: DatasetState | null;
  cleaned: CleanedDatasetResult | null;
  targetColumn: string | null;
  constraintsText: string;
  guidedMode: boolean;
  onNext: () => void;
}> = ({ dataset, cleaned, targetColumn, constraintsText, guidedMode, onNext }) => {
  const rows = cleaned?.allCleanRows ?? dataset?.rawRows ?? [];
  const columns = dataset?.columns ?? [];
  const numericCols = columns.filter(
    (c) => (c.type === "numeric_continuous" || c.type === "numeric_discrete") && !c.isConstant
  );
  const dateCols = columns.filter((c) => c.type === "temporal");

  const [corrMethod, setCorrMethod] = useState<CorrelationMethod>("pearson");
  const [selectedVar, setSelectedVar] = useState<string>(numericCols[0]?.name ?? "");
  const [scatterX, setScatterX] = useState<string>(numericCols[0]?.name ?? "");
  const [scatterY, setScatterY] = useState<string>(
    targetColumn && numericCols.some((c) => c.name === targetColumn)
      ? targetColumn
      : numericCols[numericCols.length - 1]?.name ?? ""
  );

  const corrMatrix = useMemo(
    () =>
      computeCorrelationMatrix(
        rows,
        numericCols.map((c) => c.name),
        corrMethod
      ),
    [rows, numericCols, corrMethod]
  );

  const catAssociations = useMemo(
    () => computeCategoricalAssociations(rows, columns, targetColumn),
    [rows, columns, targetColumn]
  );

  const vifEntries = useMemo(() => {
    const preds = numericCols.map((c) => c.name).filter((n) => n !== targetColumn);
    return computeVIF(rows, preds);
  }, [rows, numericCols, targetColumn]);

  const activeVar = numericCols.find((c) => c.name === selectedVar) ?? numericCols[0];
  const activeVals = useMemo(() => {
    if (!activeVar) return [];
    return rows
      .map((r) => r[activeVar.name])
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  }, [rows, activeVar]);

  const histData = useMemo(() => buildHistogramBins(activeVals, 12), [activeVals]);
  const activeBox = useMemo(() => {
    if (activeVals.length < 4) return null;
    const sorted = [...activeVals].sort((a, b) => a - b);
    const bounds = iqrBounds(activeVals);
    const outliers = activeVals.filter((v) => v < bounds.low || v > bounds.high);
    const m = activeVals.reduce((a, b) => a + b, 0) / activeVals.length;
    const med = sorted[Math.floor(sorted.length / 2)];
    return {
      min: sorted[0],
      q1: bounds.q1,
      median: med,
      q3: bounds.q3,
      max: sorted[sorted.length - 1],
      mean: m,
      outliers,
    };
  }, [activeVals]);

  const scatterPts = useMemo(() => {
    if (!scatterX || !scatterY) return [];
    const { x, y } = getNumericValuesPair(rows, scatterX, scatterY);
    return x.map((xv, i) => ({ x: xv, y: y[i] }));
  }, [rows, scatterX, scatterY]);

  const timeSeriesPts = useMemo(() => {
    if (dateCols.length === 0 || !activeVar) return [];
    return buildTimeSeriesPoints(rows, dateCols[0].name, activeVar.name);
  }, [rows, dateCols, activeVar]);

  if (!dataset || rows.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Carga y limpia una base de datos antes de explorar.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">04. Exploración Estadística (EDA)</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Distribuciones, Asociaciones y Multicolinealidad (VIF)
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Analiza distribuciones, atípicos y relaciones estadísticas en el contexto de tus restricciones
              empresariales. Recuerda: la correlación no implica causalidad.
            </p>
          )}
        </div>
        <button
          onClick={onNext}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
        >
          Ir a Selección de Variables
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Methodological reminder */}
      <div className="bg-slate-100 border border-slate-200 rounded-lg px-4 py-3 flex items-start gap-2.5 text-xs text-slate-700">
        <AlertCircle className="w-4 h-4 text-[#0B1F3A] shrink-0 mt-0.5" />
        <div>
          <strong>Regla de interpretación empresarial:</strong> Una correlación elevada describe co-movimiento
          lineal o monótono, pero <strong>no demuestra causalidad</strong>. Evalúa cada relación frente a las
          restricciones declaradas ({constraintsText || "sin restricciones registradas"}).
        </div>
      </div>

      {/* Distribution + Boxplot + Scatter */}
      {activeVar && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-[#0B1F3A]">Distribución y Boxplot Univariado</h3>
              <select
                value={activeVar.name}
                onChange={(e) => setSelectedVar(e.target.value)}
                className="border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs bg-white"
              >
                {numericCols.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>
            <HistogramChart data={histData} />
            {activeBox && (
              <BoxplotSvg
                label={activeVar.name}
                min={activeBox.min}
                q1={activeBox.q1}
                median={activeBox.median}
                q3={activeBox.q3}
                max={activeBox.max}
                mean={activeBox.mean}
                outliers={activeBox.outliers}
              />
            )}
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-[#0B1F3A]">Gráfico de Dispersión bivariado</h3>
              <div className="flex items-center gap-2 text-xs">
                <select
                  value={scatterX}
                  onChange={(e) => setScatterX(e.target.value)}
                  className="border border-slate-200 rounded-lg px-2 py-1 bg-white"
                >
                  {numericCols.map((c) => (
                    <option key={c.name} value={c.name}>
                      X: {c.name}
                    </option>
                  ))}
                </select>
                <select
                  value={scatterY}
                  onChange={(e) => setScatterY(e.target.value)}
                  className="border border-slate-200 rounded-lg px-2 py-1 bg-white"
                >
                  {numericCols.map((c) => (
                    <option key={c.name} value={c.name}>
                      Y: {c.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <ScatterTrendChart points={scatterPts} xLabel={scatterX} yLabel={scatterY} />
            {timeSeriesPts.length > 0 && (
              <div className="pt-2 border-t border-slate-100">
                <p className="text-xs font-semibold text-slate-700 mb-2">
                  Evolución Temporal de {activeVar.name} ({dateCols[0].name})
                </p>
                <TimeTrendChart data={timeSeriesPts} metricLabel={activeVar.name} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Correlation Matrix + Categorical Associations + VIF */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-[#0B1F3A]">Matriz de Correlación Numérica</h3>
              <p className="text-xs text-slate-500">
                Selecciona Pearson (lineal), Spearman (monótona/asimetria) o Kendall (muestras pequeñas/empates).
              </p>
            </div>
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              {(["pearson", "spearman", "kendall"] as CorrelationMethod[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setCorrMethod(m)}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md capitalize transition-colors ${
                    corrMethod === m ? "bg-white text-[#0B1F3A] shadow-xs" : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>
          <CorrelationHeatmap corr={corrMatrix} />
        </div>

        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-[#0B1F3A]">
              Diagnóstico de Multicolinealidad (VIF)
            </h3>
            <p className="text-xs text-slate-500">
              VIF ≥ 10 indica colinealidad severa; puedes conservar la variable si existe justificación económica o usar Ridge.
            </p>
          </div>
          <div className="space-y-2">
            {vifEntries.map((v) => (
              <div
                key={v.column}
                className="flex items-center justify-between p-2.5 rounded-lg border border-slate-200 text-xs"
              >
                <span className="font-medium text-slate-800">{v.column}</span>
                <div className="flex items-center gap-3 font-mono tabular-nums">
                  <span className="text-slate-500">Tol: {v.tolerance.toFixed(2)}</span>
                  <span
                    className={`font-semibold ${
                      v.severity === "severe"
                        ? "text-red-700"
                        : v.severity === "moderate"
                        ? "text-amber-700"
                        : "text-teal-700"
                    }`}
                  >
                    VIF: {v.vif.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {catAssociations.length > 0 && (
            <div className="pt-3 border-t border-slate-100 space-y-2">
              <h4 className="text-xs font-semibold text-[#0B1F3A]">Asociaciones con Variables Categóricas</h4>
              {catAssociations.map((a, i) => (
                <div key={i} className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                  <div className="flex items-center justify-between font-medium text-slate-900">
                    <span>
                      {a.colA} ↔ {a.colB}
                    </span>
                    <span className="font-mono text-teal-700 tabular-nums">
                      {a.metric}: {a.value.toFixed(3)}
                    </span>
                  </div>
                  <p className="text-slate-600 mt-0.5">{a.interpretation}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Descriptive Stats Table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-[#0B1F3A]">Estadísticos Descriptivos</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3 font-medium">Variable</th>
                <th className="py-2.5 px-3 font-medium text-right">N</th>
                <th className="py-2.5 px-3 font-medium text-right">Media</th>
                <th className="py-2.5 px-3 font-medium text-right">Mediana</th>
                <th className="py-2.5 px-3 font-medium text-right">Desv. Est.</th>
                <th className="py-2.5 px-3 font-medium text-right">Mín</th>
                <th className="py-2.5 px-3 font-medium text-right">Q1</th>
                <th className="py-2.5 px-3 font-medium text-right">Q3</th>
                <th className="py-2.5 px-3 font-medium text-right">Máx</th>
                <th className="py-2.5 px-3 font-medium text-right">Asimetría</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {numericCols.map((c) => {
                const s = c.stats;
                if (!s) return null;
                return (
                  <tr key={c.name} className="hover:bg-slate-50/50">
                    <td className="py-2 px-3 font-semibold text-[#0B1F3A]">{c.name}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{s.n}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{formatNum(s.mean, 2)}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{formatNum(s.median, 2)}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{formatNum(s.std, 2)}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{formatNum(s.min, 2)}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{formatNum(s.q1, 2)}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{formatNum(s.q3, 2)}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{formatNum(s.max, 2)}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">{formatNum(s.skewness, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {vifEntries.some((v) => v.severity === "severe") && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3 text-xs text-amber-900">
          <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
          <div>
            <strong>Alerta de colinealidad detectada:</strong> Existen predictores con VIF ≥ 10. Puedes depurar uno de
            los predictores redundantes en el siguiente paso o conservarlos evaluando Ridge (L2) en el comparador de modelos.
          </div>
        </div>
      )}
    </div>
  );
};
