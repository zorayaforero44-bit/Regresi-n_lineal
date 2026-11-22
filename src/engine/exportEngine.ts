import * as XLSX from "xlsx";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";
import {
  BusinessContext,
  CleanedDatasetResult,
  DataRow,
  GeminiExecutiveNarrative,
  GeminiScenarioRecommendation,
  ModelSelectionResult,
  ScenarioResult,
  SimulatedRowRecord,
  TrainedModel,
  VariableSimulationPlan,
} from "../types/analytics";
import { formatNum, formatPValue } from "./mathStats";

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function exportRowsToCsv(rows: DataRow[], filename: string) {
  if (rows.length === 0) return;
  const ws = XLSX.utils.json_to_sheet(rows);
  const csv = XLSX.utils.sheet_to_csv(ws);
  downloadBlob(new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8;" }), filename);
}

export function exportScenariosToCsv(scenarios: ScenarioResult[], targetName: string, filename: string) {
  const flat = scenarios.map((s) => ({
    Escenario: s.name,
    Tipo: s.type,
    [`Y_Estimado_${targetName}`]: Number(s.predictedY.toFixed(4)),
    Diferencia_vs_Base: Number(s.diffVsBase.toFixed(4)),
    Variacion_Pct: Number(s.pctVsBase.toFixed(2)),
    Intervalo_Inferior: Number(s.piLow.toFixed(4)),
    Intervalo_Superior: Number(s.piHigh.toFixed(4)),
    Advertencias: s.outOfRangeWarnings.join(" | ") || "Ninguna",
  }));
  exportRowsToCsv(flat, filename);
}

export function exportFullExcelWorkbook(params: {
  datasetName: string;
  cleaned: CleanedDatasetResult;
  models: TrainedModel[];
  selectedModel: TrainedModel;
  selection: ModelSelectionResult;
  simPlans: VariableSimulationPlan[];
  simRows: SimulatedRowRecord[];
  scenarios: ScenarioResult[];
  recommendations: Record<string, GeminiScenarioRecommendation>;
}) {
  const wb = XLSX.utils.book_new();

  // 1. Clean Data
  const wsClean = XLSX.utils.json_to_sheet(params.cleaned.allCleanRows);
  XLSX.utils.book_append_sheet(wb, wsClean, "1_Base_Limpia");

  // 2. Model Comparison
  const compRows = params.selection.comparisonRows.map((r) => ({
    Modelo: r.name,
    Metodo: r.method,
    CR2_Ajustado: Number(r.adjR2.toFixed(4)),
    RMSE_Validacion: Number(r.rmseVal.toFixed(4)),
    RMSE_CV: Number(r.cvRmse.toFixed(4)),
    MAE_Validacion: Number(r.maeVal.toFixed(4)),
    Significancia_t: r.significantCoefRatio,
    Valor_P_F: r.fPValue !== null ? Number(r.fPValue.toFixed(5)) : "N/A",
    Max_VIF: Number(r.maxVif.toFixed(2)),
    Diagnostico_Residuos: r.residualHealth,
    Interpretabilidad: r.interpretability,
    Observaciones: r.priorityNotes,
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(compRows), "2_Comparacion_Modelos");

  // 3. Coefficients of Selected Model
  const coefRows = params.selectedModel.coefficients.map((c) => ({
    Termino: c.term,
    Variable_Origen: c.sourceColumn,
    Coeficiente_Beta: Number(c.estimate.toFixed(6)),
    Error_Estandar: c.stdError !== null ? Number(c.stdError.toFixed(6)) : "N/A",
    Estadistico_t: c.tStat !== null ? Number(c.tStat.toFixed(4)) : "N/A",
    Valor_p: c.pValue !== null ? Number(c.pValue.toFixed(5)) : "N/A",
    IC_95_Inferior: c.ciLow !== null ? Number(c.ciLow.toFixed(4)) : "N/A",
    IC_95_Superior: c.ciHigh !== null ? Number(c.ciHigh.toFixed(4)) : "N/A",
    VIF: c.vif !== null ? Number(c.vif.toFixed(2)) : "N/A",
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(coefRows), "3_Coeficientes");

  // 4. Residual Diagnostics
  const diagRows = [
    { Prueba: "Durbin-Watson", Estadistico: Number(params.selectedModel.diagnostics.durbinWatson.toFixed(4)), Valor_p: "N/A", Nota: params.selectedModel.diagnostics.durbinWatsonNote },
    { Prueba: "Breusch-Pagan (Homocedasticidad)", Estadistico: Number(params.selectedModel.diagnostics.breuschPaganStat.toFixed(4)), Valor_p: Number(params.selectedModel.diagnostics.breuschPaganPValue.toFixed(5)), Nota: params.selectedModel.diagnostics.breuschPaganNote },
    { Prueba: "Jarque-Bera (Normalidad)", Estadistico: Number(params.selectedModel.diagnostics.jarqueBeraStat.toFixed(4)), Valor_p: Number(params.selectedModel.diagnostics.jarqueBeraPValue.toFixed(5)), Nota: params.selectedModel.diagnostics.jarqueBeraNote },
  ];
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(diagRows), "4_Diagnosticos");

  // 5. Simulation Plans
  const planRows = params.simPlans.map((p) => ({
    Variable: p.column,
    Tipo: p.type,
    Controlable: p.controllable ? "Sí" : "No",
    Metodo_Recomendado: p.recommendedStrategy,
    Metodo_Seleccionado: p.selectedStrategy,
    Calidad_Auxiliar: p.auxFitQuality,
    Justificacion: p.justification,
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(planRows), "5_Motor_Simulacion");

  // 6. Simulated Records
  const simFlat = params.simRows.map((r) => ({
    Periodo: r.periodIndex,
    Origen: r.sourceType,
    ...r.values,
    [`Y_Estimado_${params.selectedModel.targetColumn}`]: Number(r.predictedY.toFixed(2)),
    IC_95_Inf: Number(r.piLow.toFixed(2)),
    IC_95_Sup: Number(r.piHigh.toFixed(2)),
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(simFlat), "6_Datos_Simulados");

  // 7. Scenarios
  const scenRows = params.scenarios.map((s) => {
    const row: Record<string, string | number> = {
      Escenario: s.name,
      Y_Estimado: Number(s.predictedY.toFixed(2)),
      Dif_vs_Base: Number(s.diffVsBase.toFixed(2)),
      Var_Pct: Number(s.pctVsBase.toFixed(2)),
      Limite_Inf: Number(s.piLow.toFixed(2)),
      Limite_Sup: Number(s.piHigh.toFixed(2)),
    };
    for (const inp of s.inputs) row[`Input_${inp.column}`] = inp.scenarioValue;
    return row;
  });
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(scenRows), "7_Escenarios");

  // 8. Recommendations
  const recList = Object.values(params.recommendations).map((rec) => ({
    Escenario: rec.scenarioName,
    Nivel_Confianza: rec.nivelConfianza,
    Hallazgo: rec.hallazgo,
    Interpretacion: rec.interpretacion,
    Recomendacion: rec.recomendacion,
    Impacto_Esperado: rec.impactoEsperado,
    Riesgos: rec.riesgos,
    Distincion_Causal: rec.distincionCausal,
  }));
  if (recList.length > 0) {
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(recList), "8_Recomendaciones");
  }

  const safeName = params.datasetName.replace(/\.[^.]+$/, "");
  XLSX.writeFile(wb, `LRI_${safeName}_ZorayaForero.xlsx`);
}

export function exportExecutivePdfReport(params: {
  datasetName: string;
  businessContext: BusinessContext;
  cleaned: CleanedDatasetResult;
  selectedModel: TrainedModel;
  selection: ModelSelectionResult;
  scenarios: ScenarioResult[];
  recommendations: Record<string, GeminiScenarioRecommendation>;
  narrative?: GeminiExecutiveNarrative | null;
}) {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const navy: [number, number, number] = [11, 31, 58];
  const teal: [number, number, number] = [13, 148, 136];

  // Header Banner
  doc.setFillColor(...navy);
  doc.rect(0, 0, 210, 32, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(17);
  doc.setFont("helvetica", "bold");
  doc.text("Linear Regression Intelligence", 14, 14);
  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.text("Predictive Analytics & Business Decision Simulator — By Zoraya Forero", 14, 21);
  doc.setFontSize(8.5);
  doc.setTextColor(180, 225, 225);
  doc.text("Economista | Análisis de datos y automatización de procesos | Análisis financiero | Reportes para toma de decisiones", 14, 27);

  let y = 40;
  doc.setTextColor(15, 23, 42);
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("1. Pregunta de Negocio y Alcance", 14, y);
  y += 6;
  doc.setFontSize(9.5);
  doc.setFont("helvetica", "normal");
  const ctxLines = doc.splitTextToSize(
    `Sector: ${params.businessContext.sector || "No especificado"} | Base: ${params.datasetName}\n` +
      `Pregunta empresarial: ${params.businessContext.businessQuestion || "No especificada"}\n` +
      `Variable objetivo (Y): ${params.selectedModel.targetColumn} | Horizonte: ${params.businessContext.horizon || "No especificado"}\n` +
      `Restricciones: ${params.businessContext.constraints || "Ninguna especificada"}`,
    182
  );
  doc.text(ctxLines, 14, y);
  y += ctxLines.length * 4.5 + 4;

  // Section 2: Data & Anti-Leakage Summary
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("2. Calidad de Datos y Prevención de Fuga (Split-First)", 14, y);
  y += 5;
  doc.setFontSize(9.5);
  doc.setFont("helvetica", "normal");
  const splitNote = doc.splitTextToSize(
    `Registros analizados: ${params.cleaned.allCleanRows.length} (Train: ${params.cleaned.trainRows.length}, Validación: ${params.cleaned.valRows.length}, Prueba: ${params.cleaned.testRows.length}). ` +
      `Completitud: ${params.cleaned.completenessBefore.toFixed(1)}% -> ${params.cleaned.completenessAfter.toFixed(1)}%. ${params.cleaned.learnedParametersNote}`,
    182
  );
  doc.text(splitNote, 14, y);
  y += splitNote.length * 4.5 + 4;

  // Section 3: Model Comparison Table
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("3. Comparación Multicriterio de Modelos", 14, y);
  y += 3;

  autoTable(doc, {
    startY: y,
    head: [["Modelo", "CR²", "RMSE Val", "RMSE CV", "Significancia t", "VIF Máx"]],
    body: params.selection.comparisonRows.map((r) => [
      r.name,
      `${(r.adjR2 * 100).toFixed(1)}%`,
      formatNum(r.rmseVal, 2),
      formatNum(r.cvRmse, 2),
      r.significantCoefRatio,
      formatNum(r.maxVif, 2),
    ]),
    styles: { fontSize: 8, cellPadding: 2 },
    headStyles: { fillColor: navy, textColor: 255 },
  });

  y = ((doc as any).lastAutoTable?.finalY ?? y + 30) + 8;

  // Section 4: Selected Model & Coefficients
  if (y > 230) {
    doc.addPage();
    y = 16;
  }
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text(`4. Modelo Seleccionado: ${params.selectedModel.name}`, 14, y);
  y += 5;
  doc.setFontSize(8.5);
  doc.setFont("courier", "normal");
  const eqLines = doc.splitTextToSize(params.selectedModel.equation, 182);
  doc.text(eqLines, 14, y);
  y += eqLines.length * 4 + 3;

  autoTable(doc, {
    startY: y,
    head: [["Término", "Coef. β", "Error Est.", "t", "p-valor", "IC 95%"]],
    body: params.selectedModel.coefficients.map((c) => [
      c.term,
      formatNum(c.estimate, 4),
      c.stdError !== null ? formatNum(c.stdError, 4) : "N/A",
      c.tStat !== null ? formatNum(c.tStat, 3) : "N/A",
      formatPValue(c.pValue),
      c.ciLow !== null && c.ciHigh !== null ? `[${formatNum(c.ciLow, 2)}, ${formatNum(c.ciHigh, 2)}]` : "N/A",
    ]),
    styles: { fontSize: 8, cellPadding: 2 },
    headStyles: { fillColor: teal, textColor: 255 },
  });

  y = ((doc as any).lastAutoTable?.finalY ?? y + 30) + 8;

  // Section 5: Scenarios Table
  if (y > 220) {
    doc.addPage();
    y = 16;
  }
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("5. Laboratorio de Escenarios y Simulación", 14, y);
  y += 3;

  autoTable(doc, {
    startY: y,
    head: [["Escenario", "Y Estimado", "Dif. vs Base", "Var. %", "Rango Incertidumbre"]],
    body: params.scenarios.map((s) => [
      s.name,
      formatNum(s.predictedY, 2),
      formatNum(s.diffVsBase, 2),
      `${s.pctVsBase >= 0 ? "+" : ""}${s.pctVsBase.toFixed(1)}%`,
      `[${formatNum(s.piLow, 2)}, ${formatNum(s.piHigh, 2)}]`,
    ]),
    styles: { fontSize: 8, cellPadding: 2 },
    headStyles: { fillColor: navy, textColor: 255 },
  });

  y = ((doc as any).lastAutoTable?.finalY ?? y + 30) + 8;

  // Section 6: Executive Narrative & Recommendations
  const recEntries = Object.values(params.recommendations);
  if (params.narrative || recEntries.length > 0) {
    if (y > 220) {
      doc.addPage();
      y = 16;
    }
    doc.setFontSize(12);
    doc.setFont("helvetica", "bold");
    doc.text("6. Recomendaciones Estratégicas y Conclusiones", 14, y);
    y += 5;
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");

    if (params.narrative) {
      const sumLines = doc.splitTextToSize(params.narrative.executiveSummary, 182);
      doc.text(sumLines, 14, y);
      y += sumLines.length * 4.2 + 4;
    }

    for (const rec of recEntries.slice(0, 3)) {
      if (y > 245) {
        doc.addPage();
        y = 16;
      }
      doc.setFont("helvetica", "bold");
      doc.text(`${rec.scenarioName} (${rec.nivelConfianza})`, 14, y);
      y += 4.5;
      doc.setFont("helvetica", "normal");
      const block = doc.splitTextToSize(
        `Hallazgo: ${rec.hallazgo}\nRecomendación: ${rec.recomendacion}\nImpacto esperado: ${rec.impactoEsperado}\nRiesgos: ${rec.riesgos}`,
        182
      );
      doc.text(block, 14, y);
      y += block.length * 4 + 4;
    }
  }

  // Footer & Academic Credits on last page
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(7.5);
    doc.setTextColor(100, 116, 139);
    doc.text(
      `Linear Regression Intelligence — By Zoraya Forero | Página ${i} de ${pageCount}`,
      14,
      289
    );
    if (i === pageCount) {
      doc.text(
        "Desarrollado a partir de conceptos académicos de regresión y pronósticos empresariales abordados en clase por el profesor Hugo Alexander Martínez Gallego,",
        14,
        280
      );
      doc.text(
        "complementados con documentación estadística y metodologías de analítica de datos (NIST, Penn State, statsmodels, scikit-learn, SciPy).",
        14,
        284
      );
    }
  }

  const safeName = params.datasetName.replace(/\.[^.]+$/, "");
  doc.save(`Informe_Ejecutivo_${safeName}_ZorayaForero.pdf`);
}
