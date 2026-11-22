# Linear Regression Intelligence

**Predictive Analytics & Business Decision Simulator — By Zoraya Forero**  
*Economista | Análisis de datos y automatización de procesos | Análisis financiero | Reportes para la toma de decisiones.*

## Capacidades Principales
1. **Carga Universal y Diagnóstico**: Lectura real de CSV, XLSX, XLS, JSON y TSV, visualización inmediata de las primeras 5 filas reales, auditoría de calidad y diccionario de datos editable.
2. **Contextualización Empresarial con Gemini**: Análisis del problema económico y clasificación de variables objetivo (Y) y predictores (X).
3. **Limpieza y Split Anti-Fuga**: Separación Train/Validación/Prueba (aleatoria o cronológica) ejecutada **antes** de ajustar imputadores (media, mediana, KNN, grupos, moda, arrastre temporal pasado-solo) y cotas IQR.
4. **Exploración Estadística (EDA)**: Correlaciones Pearson/Spearman/Kendall, Cramér's V, Eta-cuadrado ($\eta^2$), VIF, boxplots, dispersión y tendencias temporales.
5. **Modelado y Comparación Multicriterio**: OLS Simple, OLS Múltiple, Ridge (L2) y Lasso (L1) complementarios, validación cruzada K-Fold y temporal expansiva, diagnósticos Durbin-Watson / Breusch-Pagan / Jarque-Bera, y comparador transparente centrado en $CR^2$, RMSE fuera de muestra y pruebas $t$.
6. **Motor Estadístico Individual de Simulación y Laboratorio de Escenarios**: Estrategia diferenciada por variable, 6 escenarios obligatorios (Base, Optimista, Conservador, Adverso, Personalizado, Probabilístico Monte Carlo), intervalos de predicción 95% y sensibilidad Tornado.
7. **Recomendaciones IA y Exportación**: Recomendaciones estructuradas con Gemini, libro Excel completo (`.xlsx`), CSV e informe ejecutivo en PDF.

## Despliegue en Vercel y Ejecución Local

### Variables de Entorno
Configura en `.env` (local) o en **Project Settings → Environment Variables** en Vercel:
```env
GEMINI_API_KEY="tu_clave_gemini"
```

### Ejecución Local
```bash
npm install
npm run dev
```
La aplicación se ejecuta en `http://localhost:3000`.

### Despliegue en Vercel
1. Sube el repositorio a GitHub.
2. Importa el repositorio en Vercel (`Framework Preset: Vite`, `Build Command: npm run build`, `Output Directory: dist`).
3. Agrega la variable de entorno `GEMINI_API_KEY` en Vercel.
4. Las rutas serverless `/api/gemini/analyze-context`, `/api/gemini/recommend` y `/api/gemini/executive-summary` quedan operativas automáticamente mediante `vercel.json`.

## Créditos Académicos
Desarrollado a partir de conceptos académicos de regresión y pronósticos empresariales abordados en clase por el profesor Hugo Alexander Martínez Gallego, complementados con documentación estadística y metodologías de analítica de datos (NIST/SEMATECH, Penn State, statsmodels, scikit-learn y SciPy).
