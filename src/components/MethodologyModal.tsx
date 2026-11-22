import React from "react";
import { X, BookOpen, ShieldCheck, Layers } from "lucide-react";

export const MethodologyModal: React.FC<{
  open: boolean;
  onClose: () => void;
}> = ({ open, onClose }) => {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 backdrop-blur-xs p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="methodology-title"
    >
      <div className="bg-white border border-slate-200 rounded-xl max-w-3xl w-full max-h-[88vh] overflow-y-auto shadow-xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-[#0B1F3A] text-white rounded-t-xl">
          <div>
            <h2 id="methodology-title" className="text-lg font-bold tracking-tight">
              Fundamento Académico y Metodología Estadística
            </h2>
            <p className="text-xs text-teal-200 mt-0.5">
              Linear Regression Intelligence — By Zoraya Forero
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-slate-300 hover:text-white hover:bg-white/10 transition-colors"
            aria-label="Cerrar ventana"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6 text-sm text-slate-700 leading-relaxed">
          <section className="space-y-2">
            <div className="flex items-center gap-2 text-[#0B1F3A] font-semibold">
              <BookOpen className="w-4 h-4 text-teal-600" />
              <h3>1. Conceptos Académicos del Curso</h3>
            </div>
            <p>
              Desarrollado a partir de conceptos académicos de regresión y pronósticos empresariales abordados en
              clase por el profesor <strong>Hugo Alexander Martínez Gallego</strong>, complementados con
              documentación estadística y metodologías de analítica de datos.
            </p>
            <ul className="list-disc pl-5 space-y-1 text-slate-600">
              <li>
                <strong>Pruebas de hipótesis sobre parámetros:</strong> Evaluación de significancia individual{" "}
                <span className="font-mono text-xs">H₀: βᵢ = 0</span> vs{" "}
                <span className="font-mono text-xs">H₁: βᵢ ≠ 0</span> mediante estadístico t y p-valores.
              </li>
              <li>
                <strong>Esquemas de validación rigurosa:</strong> Separación explícita en conjuntos de
                Entrenamiento, Validación y Prueba, junto con Validación Cruzada K-Fold para prevenir sobreajuste.
              </li>
              <li>
                <strong>Uso de variables escalares, ordinales y dummies:</strong> Codificación correcta con categoría
                de referencia y supuestos de medición representativa.
              </li>
              <li>
                <strong>Criterio de selección en regresión lineal vs logística:</strong> Cuando la variable dependiente
                Y es binaria/dicotómica (0/1), la plataforma advierte que no corresponde aplicar regresión lineal
                clásica y orienta hacia regresión logística.
              </li>
            </ul>
          </section>

          <hr className="border-slate-200" />

          <section className="space-y-2">
            <div className="flex items-center gap-2 text-[#0B1F3A] font-semibold">
              <Layers className="w-4 h-4 text-teal-600" />
              <h3>2. Ampliaciones Metodológicas de Fuentes Científicas</h3>
            </div>
            <p className="text-slate-600">
              Las implementaciones numéricas exactas se apoyan en estándares documentados en{" "}
              <strong>NIST/SEMATECH e-Handbook</strong>, <strong>Penn State STAT 462/501</strong>,{" "}
              <strong>statsmodels</strong>, <strong>scikit-learn</strong> y <strong>SciPy</strong>:
            </p>
            <ul className="list-disc pl-5 space-y-1 text-slate-600">
              <li>
                <strong>Inferencia exacta OLS:</strong> Estimación matricial{" "}
                <span className="font-mono text-xs">(XᵀX)⁻¹Xᵀy</span>, funciones Beta y Gamma incompletas
                regularizadas para p-valores exactos t, F y χ², e intervalos de predicción{" "}
                <span className="font-mono text-xs">ŷ ± t · s · √(1 + x₀ᵀ(XᵀX)⁻¹x₀)</span>.
              </li>
              <li>
                <strong>Diagnósticos de supuestos:</strong> Factor de Inflación de Varianza (VIF), Durbin-Watson
                (autocorrelación), Breusch-Pagan (homocedasticidad) y Jarque-Bera (normalidad de residuos).
              </li>
              <li>
                <strong>Regularización diferenciada:</strong> Ridge (L2) y Lasso (L1) con estandarización ajustada
                estrictamente en entrenamiento y sin atribuir p-valores clásicos de OLS.
              </li>
              <li>
                <strong>Prevención de fuga de información (Data Leakage):</strong> Split previo al ajuste de
                imputadores (media, mediana, KNN, grupos) y validación temporal expansiva cuando existen fechas.
              </li>
            </ul>
          </section>

          <hr className="border-slate-200" />

          <section className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-1.5">
            <div className="flex items-center gap-2 text-[#0B1F3A] font-semibold text-xs">
              <ShieldCheck className="w-4 h-4 text-teal-600" />
              <span>Autoría y Propiedad Intelectual del Software</span>
            </div>
            <p className="text-xs text-slate-600">
              <strong>Linear Regression Intelligence</strong> es un desarrollo de software independiente creado por{" "}
              <strong>Zoraya Forero</strong> (Economista · Análisis de datos, automatización de procesos y reportes
              para toma de decisiones). Los conceptos académicos citados se reconocen exclusivamente como base
              formativa, sin atribuir al profesor la propiedad ni el desarrollo de esta plataforma.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};
