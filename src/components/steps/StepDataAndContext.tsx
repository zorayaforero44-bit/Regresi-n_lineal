import React, { useRef, useState } from "react";
import {
  Upload,
  FileSpreadsheet,
  AlertTriangle,
  Sparkles,
  CheckCircle2,
  ArrowRight,
  Database,
} from "lucide-react";
import {
  BusinessContext,
  BusinessRole,
  ColumnProfile,
  DatasetState,
  GeminiContextAnalysis,
  VariableType,
} from "../../types/analytics";
import { SampleDatasetSpec } from "../../engine/dataParser";
import { formatNum } from "../../engine/mathStats";

const TYPE_LABELS: Record<VariableType, string> = {
  numeric_continuous: "Numérica continua",
  numeric_discrete: "Numérica discreta",
  categorical: "Categórica",
  ordinal: "Ordinal",
  binary: "Binaria",
  temporal: "Temporal / Fecha",
  textual: "Textual",
};

const ROLE_LABELS: Record<BusinessRole, string> = {
  controllable: "Modificable por el negocio",
  external_fixed: "Externa / Estructural",
  identifier_exclude: "Identificador / Excluir",
};

export const StepLoadAndDictionary: React.FC<{
  dataset: DatasetState | null;
  samples: SampleDatasetSpec[];
  guidedMode: boolean;
  onFileUpload: (file: File) => Promise<void>;
  onLoadSample: (sample: SampleDatasetSpec) => void;
  onUpdateColumn: (
    colName: string,
    patch: Partial<Pick<ColumnProfile, "type" | "unit" | "description" | "businessRole">>
  ) => void;
  onNext: () => void;
}> = ({ dataset, samples, guidedMode, onFileUpload, onLoadSample, onUpdateColumn, onNext }) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleFiles = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    setLoading(true);
    setUploadError(null);
    try {
      await onFileUpload(file);
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : "No fue posible leer el archivo.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">01. Carga Universal y Diccionario de Datos</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Ingesta de Datos y Diagnóstico Estructural
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Carga un archivo CSV, XLSX, XLS, JSON o TSV (o selecciona una base empresarial de ejemplo). Verificaremos
              al instante las primeras 5 filas reales, los tipos de variables y los problemas iniciales de calidad.
            </p>
          )}
        </div>
        {dataset && dataset.rawRows.length > 0 && (
          <button
            onClick={onNext}
            className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Continuar a Contexto de Negocio
            <ArrowRight className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Upload + Sample Datasets */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={`lg:col-span-5 bg-white border rounded-xl p-6 flex flex-col justify-between transition-colors ${
            dragging ? "border-teal-600 bg-teal-50/40" : "border-slate-200"
          }`}
        >
          <div>
            <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center text-[#0B1F3A] mb-4">
              <Upload className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-[#0B1F3A]">Cargar Archivo Propio</h3>
            <p className="text-sm text-slate-600 mt-1">
              Compatible con <strong>CSV, XLSX, XLS, JSON y TSV</strong>. Detección automática de formatos
              numéricos con símbolos, fechas y archivos Parquet o no estructurados.
            </p>
          </div>

          <div className="mt-6 space-y-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls,.json,.tsv,.txt,.parquet"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="w-full py-2.5 px-4 bg-teal-600 hover:bg-teal-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
            >
              {loading ? "Procesando archivo..." : "Seleccionar archivo desde el equipo"}
            </button>
            <p className="text-xs text-slate-500 text-center">
              O arrastra y suelta el archivo en esta zona · Procesamiento numérico verificable
            </p>
            {uploadError && (
              <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700">
                {uploadError}
              </div>
            )}
          </div>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 text-[#0B1F3A] mb-3">
              <Database className="w-4 h-4 text-teal-600" />
              <h3 className="text-base font-semibold">Bases Empresariales de Prueba</h3>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Datasets reales simulados con valores faltantes, duplicados y variantes ortográficas para auditar todo el recorrido estadístico.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {samples.map((s) => {
                const isActive = dataset?.fileName === `${s.id}.csv`;
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => onLoadSample(s)}
                    className={`text-left p-4 rounded-lg border transition-colors flex flex-col justify-between ${
                      isActive
                        ? "border-teal-600 bg-teal-50/50"
                        : "border-slate-200 hover:border-slate-300 bg-slate-50/50"
                    }`}
                  >
                    <div>
                      <p className="text-xs font-medium text-teal-700">{s.sector}</p>
                      <p className="text-sm font-semibold text-[#0B1F3A] mt-1">{s.title}</p>
                      <p className="text-xs text-slate-600 mt-1.5 leading-relaxed">{s.description}</p>
                    </div>
                    <span className="mt-3 text-xs font-medium text-[#0B1F3A] inline-flex items-center gap-1">
                      <FileSpreadsheet className="w-3.5 h-3.5 text-teal-600" />
                      {isActive ? "Base activa" : "Cargar base"}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Warning for Parquet / unstructured */}
      {dataset?.unstructuredWarning && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-semibold text-amber-900">
              Advertencia técnica sobre el archivo "{dataset.fileName}"
            </h4>
            <p className="text-sm text-amber-800 mt-1">{dataset.unstructuredWarning}</p>
          </div>
        </div>
      )}

      {/* Diagnostic Cards & First 5 Real Rows */}
      {dataset && dataset.rawRows.length > 0 && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
            <div className="bg-white border border-slate-200 rounded-lg p-3.5">
              <p className="text-xs text-slate-500">Registros</p>
              <p className="text-xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
                {dataset.rawRows.length}
              </p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3.5">
              <p className="text-xs text-slate-500">Columnas</p>
              <p className="text-xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
                {dataset.columns.length}
              </p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3.5">
              <p className="text-xs text-slate-500">Numéricas</p>
              <p className="text-xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
                {
                  dataset.columns.filter(
                    (c) => c.type === "numeric_continuous" || c.type === "numeric_discrete"
                  ).length
                }
              </p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3.5">
              <p className="text-xs text-slate-500">Categ. / Binarias</p>
              <p className="text-xl font-bold text-[#0B1F3A] font-mono tabular-nums mt-1">
                {
                  dataset.columns.filter(
                    (c) => c.type === "categorical" || c.type === "ordinal" || c.type === "binary"
                  ).length
                }
              </p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3.5">
              <p className="text-xs text-slate-500">Celdas Faltantes</p>
              <p className="text-xl font-bold text-amber-700 font-mono tabular-nums mt-1">
                {dataset.totalMissingCells}
              </p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3.5">
              <p className="text-xs text-slate-500">Filas Duplicadas</p>
              <p className="text-xl font-bold text-amber-700 font-mono tabular-nums mt-1">
                {dataset.duplicateRowCount}
              </p>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3.5">
              <p className="text-xs text-slate-500">Completitud</p>
              <p className="text-xl font-bold text-teal-700 font-mono tabular-nums mt-1">
                {dataset.completenessPct.toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Mandatory First 5 Real Rows */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold text-[#0B1F3A]">
                  Primeras 5 Filas Reales de la Base ({dataset.fileName})
                </h3>
                <p className="text-xs text-slate-500">
                  Inspección directa sin alteraciones · Formato {dataset.fileFormat}
                </p>
              </div>
              <span className="text-xs font-mono text-slate-500 tabular-nums">
                Mostrando {dataset.firstFiveRows.length} de {dataset.rawRows.length} registros
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3 font-medium">#</th>
                    {dataset.columns.map((c) => (
                      <th key={c.name} className="py-2.5 px-3 font-medium whitespace-nowrap">
                        {c.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {dataset.firstFiveRows.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/70">
                      <td className="py-2 px-3 font-mono text-slate-400 tabular-nums">{idx + 1}</td>
                      {dataset.columns.map((c) => {
                        const val = row[c.name];
                        const isNull = val === null || val === undefined || val === "";
                        return (
                          <td
                            key={c.name}
                            className={`py-2 px-3 whitespace-nowrap tabular-nums ${
                              typeof val === "number" ? "font-mono text-right text-slate-800" : "text-slate-700"
                            }`}
                          >
                            {isNull ? <span className="text-amber-600 italic">Nulo</span> : String(val)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Quality Alerts Summary */}
          {dataset.qualityIssues.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-semibold text-[#0B1F3A]">
                Posibles Problemas de Calidad Detectados ({dataset.qualityIssues.length})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {dataset.qualityIssues.map((issue) => (
                  <div
                    key={issue.id}
                    className="p-3 rounded-lg border border-slate-200 bg-slate-50/60 flex items-start gap-2.5"
                  >
                    <AlertTriangle
                      className={`w-4 h-4 shrink-0 mt-0.5 ${
                        issue.severity === "high"
                          ? "text-red-600"
                          : issue.severity === "medium"
                          ? "text-amber-600"
                          : "text-slate-500"
                      }`}
                    />
                    <div className="text-xs">
                      <p className="font-semibold text-slate-900">{issue.title}</p>
                      <p className="text-slate-600 mt-0.5">{issue.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Editable Data Dictionary */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 border-b border-slate-200">
              <h3 className="text-sm font-semibold text-[#0B1F3A]">
                Diccionario de Datos Editable y Corrección Manual de Tipos
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Ajusta el tipo de dato detectado, unidad de medida, descripción y si la variable puede ser modificada por el negocio.
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead className="bg-slate-50 text-slate-600 border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3 font-medium">Columna</th>
                    <th className="py-2.5 px-3 font-medium">Tipo de Dato (Editable)</th>
                    <th className="py-2.5 px-3 font-medium">Unidad</th>
                    <th className="py-2.5 px-3 font-medium">Rol en Negocio</th>
                    <th className="py-2.5 px-3 font-medium">Descripción Empresarial</th>
                    <th className="py-2.5 px-3 font-medium text-right">Nulos</th>
                    <th className="py-2.5 px-3 font-medium text-right">Únicos</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {dataset.columns.map((col) => (
                    <tr key={col.name} className="hover:bg-slate-50/50">
                      <td className="py-2.5 px-3 font-semibold text-[#0B1F3A] whitespace-nowrap">
                        {col.name}
                        {col.type !== col.detectedType && (
                          <span className="block text-[11px] font-normal text-teal-700">
                            Corregido (era {TYPE_LABELS[col.detectedType]})
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3">
                        <select
                          value={col.type}
                          onChange={(e) =>
                            onUpdateColumn(col.name, { type: e.target.value as VariableType })
                          }
                          className="border border-slate-200 rounded-md px-2 py-1 text-xs bg-white text-slate-800 focus:outline-none focus:border-teal-600"
                        >
                          {Object.entries(TYPE_LABELS).map(([k, lbl]) => (
                            <option key={k} value={k}>
                              {lbl}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="text"
                          value={col.unit}
                          onChange={(e) => onUpdateColumn(col.name, { unit: e.target.value })}
                          className="w-28 border border-slate-200 rounded-md px-2 py-1 text-xs bg-white focus:outline-none focus:border-teal-600"
                        />
                      </td>
                      <td className="py-2 px-3">
                        <select
                          value={col.businessRole}
                          onChange={(e) =>
                            onUpdateColumn(col.name, { businessRole: e.target.value as BusinessRole })
                          }
                          className="border border-slate-200 rounded-md px-2 py-1 text-xs bg-white text-slate-800 focus:outline-none focus:border-teal-600"
                        >
                          {Object.entries(ROLE_LABELS).map(([k, lbl]) => (
                            <option key={k} value={k}>
                              {lbl}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="text"
                          value={col.description}
                          placeholder="Significado operativo o financiero..."
                          onChange={(e) => onUpdateColumn(col.name, { description: e.target.value })}
                          className="w-56 border border-slate-200 rounded-md px-2 py-1 text-xs bg-white focus:outline-none focus:border-teal-600"
                        />
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700">
                        {col.missingCount} ({formatNum(col.missingPct, 1)}%)
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700">
                        {col.uniqueCount}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export const StepBusinessContext: React.FC<{
  dataset: DatasetState | null;
  context: BusinessContext;
  analysis: GeminiContextAnalysis | null;
  guidedMode: boolean;
  onChangeContext: (patch: Partial<BusinessContext>) => void;
  onAnalyzeWithGemini: () => Promise<void>;
  onNext: () => void;
}> = ({ dataset, context, analysis, guidedMode, onChangeContext, onAnalyzeWithGemini, onNext }) => {
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = async () => {
    setAnalyzing(true);
    setError(null);
    try {
      await onAnalyzeWithGemini();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error analizando el contexto con Gemini.");
    } finally {
      setAnalyzing(false);
    }
  };

  if (!dataset || dataset.rawRows.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Primero carga una base de datos en el paso 01.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-700">02. Contextualización del Negocio</p>
          <h2 className="text-2xl font-bold text-[#0B1F3A] tracking-tight mt-0.5">
            Definición del Problema Empresarial y Restricciones
          </h2>
          {guidedMode && (
            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              Antes de elegir variables o entrenar regresiones, define el contexto económico y operativo. Gemini
              analizará tus respuestas junto con el diccionario de variables para orientar la selección de Y y X.
            </p>
          )}
        </div>
        <button
          onClick={onNext}
          className="inline-flex items-center gap-2 px-4 py-2 bg-[#0B1F3A] text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
        >
          Ir a Limpieza y Split
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                1. ¿A qué sector pertenece la información?
              </label>
              <input
                type="text"
                value={context.sector}
                onChange={(e) => onChangeContext({ sector: e.target.value })}
                placeholder="Ej. Retail, Banca, Manufactura, Salud, Logística..."
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-teal-600"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                2. ¿Qué representa cada registro (fila)?
              </label>
              <input
                type="text"
                value={context.recordUnit}
                onChange={(e) => onChangeContext({ recordUnit: e.target.value })}
                placeholder="Ej. Mes por sucursal, cliente, factura, planta-turno..."
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-teal-600"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              3. ¿Qué pregunta empresarial desea responder?
            </label>
            <textarea
              rows={2}
              value={context.businessQuestion}
              onChange={(e) => onChangeContext({ businessQuestion: e.target.value })}
              placeholder="Ej. ¿Cómo influyen la publicidad y el precio sobre las ventas mensuales?"
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-teal-600"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                4. ¿Qué resultado necesita explicar o pronosticar?
              </label>
              <input
                type="text"
                value={context.targetOutcome}
                onChange={(e) => onChangeContext({ targetOutcome: e.target.value })}
                placeholder="Ej. Ventas mensuales, recaudo, productividad..."
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-teal-600"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                5. ¿Cuál es el horizonte de análisis?
              </label>
              <input
                type="text"
                value={context.horizon}
                onChange={(e) => onChangeContext({ horizon: e.target.value })}
                placeholder="Ej. Próximo trimestre, 6 meses, anual..."
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-teal-600"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              6. ¿Qué variables puede modificar el negocio?
            </label>
            <input
              type="text"
              value={context.controllableVars}
              onChange={(e) => onChangeContext({ controllableVars: e.target.value })}
              placeholder="Ej. Inversión publicitaria, precio, descuento, horas de capacitación..."
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-teal-600"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              7. ¿Qué restricciones deben respetarse?
            </label>
            <textarea
              rows={2}
              value={context.constraints}
              onChange={(e) => onChangeContext({ constraints: e.target.value })}
              placeholder="Ej. Límite presupuestal, precio mínimo rentable, capacidad máxima operativa..."
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-teal-600"
            />
          </div>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={runAnalysis}
              disabled={analyzing}
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-teal-600 hover:bg-teal-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              {analyzing ? "Analizando con Gemini..." : "Analizar contexto y variables con Gemini"}
            </button>
            {error && <p className="text-xs text-red-600">{error}</p>}
          </div>
        </div>

        {/* Gemini Contextual Analysis Result */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="text-sm font-semibold text-[#0B1F3A]">Diagnóstico Contextual IA (Gemini)</h3>
            {analysis && (
              <span className="text-xs font-medium text-teal-700 inline-flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Enfoque: {analysis.analyticalFocus}
              </span>
            )}
          </div>

          {!analysis ? (
            <div className="py-12 text-center space-y-2">
              <p className="text-sm text-slate-600">
                Completa las preguntas del negocio y haz clic en <strong>"Analizar contexto y variables con Gemini"</strong>.
              </p>
              <p className="text-xs text-slate-400">
                Solo se envían metadatos y nombres de columnas; los registros confidenciales permanecen en tu sesión.
              </p>
            </div>
          ) : (
            <div className="space-y-4 text-xs leading-relaxed">
              <div>
                <p className="font-semibold text-[#0B1F3A] mb-1">Síntesis del Problema</p>
                <p className="text-slate-700">{analysis.problemSynthesis}</p>
              </div>

              <div>
                <p className="font-semibold text-[#0B1F3A] mb-1.5">Variables Objetivo (Y) Evaluadas</p>
                <div className="space-y-2">
                  {analysis.suggestedTargets.map((t) => (
                    <div key={t.column} className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                      <div className="flex items-center justify-between font-semibold text-slate-900">
                        <span>{t.column}</span>
                        <span
                          className={
                            t.suitability === "alta"
                              ? "text-teal-700"
                              : t.suitability === "no_apta_lineal"
                              ? "text-red-700"
                              : "text-amber-700"
                          }
                        >
                          Viabilidad: {t.suitability}
                        </span>
                      </div>
                      <p className="text-slate-600 mt-1">{t.reason}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <p className="font-semibold text-[#0B1F3A] mb-1.5">Clasificación Preliminar de Predictores (X)</p>
                <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1">
                  {analysis.predictorAssessments.map((p) => (
                    <div key={p.column} className="p-2 rounded border border-slate-100 bg-slate-50/70">
                      <span className="font-semibold text-slate-900">{p.column}</span>{" "}
                      <span className="text-slate-400">·</span>{" "}
                      <span
                        className={
                          p.tier === "Recomendada"
                            ? "text-teal-700 font-medium"
                            : p.tier === "No recomendada"
                            ? "text-red-700 font-medium"
                            : "text-amber-700 font-medium"
                        }
                      >
                        {p.tier}
                      </span>
                      <p className="text-slate-600 mt-0.5">{p.reason}</p>
                    </div>
                  ))}
                </div>
              </div>

              {analysis.keyConstraintsAndRisks.length > 0 && (
                <div>
                  <p className="font-semibold text-[#0B1F3A] mb-1">Restricciones y Riesgos Metodológicos</p>
                  <ul className="list-disc pl-4 space-y-1 text-slate-600">
                    {analysis.keyConstraintsAndRisks.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
