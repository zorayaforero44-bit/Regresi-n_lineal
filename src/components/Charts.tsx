import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  CoefficientStat,
  CorrelationMatrixResult,
  ScenarioResult,
  SensitivityBar,
} from "../types/analytics";
import { formatNum } from "../engine/mathStats";

export const HistogramChart: React.FC<{
  data: Array<{ label: string; count: number }>;
  color?: string;
  height?: number;
}> = ({ data, color = "#0D9488", height = 220 }) => {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10, fill: "#475569" }}
            angle={-25}
            textAnchor="end"
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fontSize: 11, fill: "#475569" }} width={38} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0B1F3A",
              border: "none",
              borderRadius: "8px",
              color: "#fff",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="count" name="Frecuencia" fill={color} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const BoxplotSvg: React.FC<{
  label: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  outliers?: number[];
}> = ({ label, min, q1, median, q3, max, mean, outliers = [] }) => {
  const allVals = [min, max, ...outliers];
  const lo = Math.min(...allVals);
  const hi = Math.max(...allVals);
  const span = hi - lo || 1;
  const toX = (v: number) => 28 + ((v - lo) / span) * 284;

  const xMin = toX(min);
  const xQ1 = toX(q1);
  const xMed = toX(median);
  const xQ3 = toX(q3);
  const xMax = toX(max);
  const xMean = toX(mean);

  return (
    <div className="w-full">
      <svg viewBox="0 0 340 82" className="w-full h-24">
        <line x1={28} y1={42} x2={312} y2={42} stroke="#f1f5f9" strokeWidth={1} />
        {/* Whiskers */}
        <line x1={xMin} y1={42} x2={xQ1} y2={42} stroke="#0B1F3A" strokeWidth={1.5} />
        <line x1={xQ3} y1={42} x2={xMax} y2={42} stroke="#0B1F3A" strokeWidth={1.5} />
        <line x1={xMin} y1={30} x2={xMin} y2={54} stroke="#0B1F3A" strokeWidth={1.5} />
        <line x1={xMax} y1={30} x2={xMax} y2={54} stroke="#0B1F3A" strokeWidth={1.5} />
        {/* IQR Box */}
        <rect
          x={xQ1}
          y={22}
          width={Math.max(4, xQ3 - xQ1)}
          height={40}
          fill="#ccfbf1"
          stroke="#0D9488"
          strokeWidth={1.5}
          rx={2}
        />
        {/* Median line */}
        <line x1={xMed} y1={22} x2={xMed} y2={62} stroke="#0B1F3A" strokeWidth={2.5} />
        {/* Mean diamond */}
        <polygon
          points={`${xMean},37 ${xMean + 4},42 ${xMean},47 ${xMean - 4},42`}
          fill="#0F766E"
        />
        {/* Outliers */}
        {outliers.slice(0, 12).map((o, idx) => (
          <circle key={idx} cx={toX(o)} cy={42} r={2.8} fill="#DC2626" opacity={0.8} />
        ))}
        {/* Labels */}
        <text x={28} y={76} fontSize={9} fill="#64748b" fontFamily="JetBrains Mono">
          Mín: {formatNum(min, 1)}
        </text>
        <text x={170} y={76} fontSize={9} fill="#0B1F3A" textAnchor="middle" fontFamily="JetBrains Mono">
          Mediana {label}: {formatNum(median, 1)}
        </text>
        <text x={312} y={76} fontSize={9} fill="#64748b" textAnchor="end" fontFamily="JetBrains Mono">
          Máx: {formatNum(max, 1)}
        </text>
      </svg>
    </div>
  );
};

export const ScatterTrendChart: React.FC<{
  points: Array<{ x: number; y: number }>;
  xLabel: string;
  yLabel: string;
}> = ({ points, xLabel, yLabel }) => {
  // Simple OLS line endpoints for visual overlay
  const n = points.length;
  let lineData: Array<{ x: number; yTrend: number }> = [];
  if (n >= 2) {
    const mx = points.reduce((a, p) => a + p.x, 0) / n;
    const my = points.reduce((a, p) => a + p.y, 0) / n;
    let num = 0;
    let den = 0;
    for (const p of points) {
      num += (p.x - mx) * (p.y - my);
      den += (p.x - mx) ** 2;
    }
    const b1 = den > 0 ? num / den : 0;
    const b0 = my - b1 * mx;
    const xs = points.map((p) => p.x);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    lineData = [
      { x: minX, yTrend: b0 + b1 * minX },
      { x: maxX, yTrend: b0 + b1 * maxX },
    ];
  }

  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <ComposedChart margin={{ top: 12, right: 16, left: 4, bottom: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            type="number"
            dataKey="x"
            name={xLabel}
            tick={{ fontSize: 11, fill: "#475569" }}
            domain={["auto", "auto"]}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={yLabel}
            tick={{ fontSize: 11, fill: "#475569" }}
            width={58}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{
              backgroundColor: "#0B1F3A",
              border: "none",
              borderRadius: "8px",
              color: "#fff",
              fontSize: "12px",
            }}
          />
          <Scatter name="Observaciones" data={points} fill="#0D9488" fillOpacity={0.72} />
          {lineData.length === 2 && (
            <Line
              data={lineData}
              dataKey="yTrend"
              dot={false}
              stroke="#0B1F3A"
              strokeWidth={2}
              name="Tendencia lineal"
              type="linear"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export const TimeTrendChart: React.FC<{
  data: Array<{ date: string; value: number }>;
  metricLabel: string;
}> = ({ data, metricLabel }) => {
  return (
    <div style={{ width: "100%", height: 240 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 16, left: 4, bottom: 18 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#475569" }} minTickGap={24} />
          <YAxis tick={{ fontSize: 11, fill: "#475569" }} width={56} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0B1F3A",
              border: "none",
              borderRadius: "8px",
              color: "#fff",
              fontSize: "12px",
            }}
          />
          <Line
            type="monotone"
            dataKey="value"
            name={metricLabel}
            stroke="#0D9488"
            strokeWidth={2}
            dot={{ r: 2.5, fill: "#0B1F3A" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export const CorrelationHeatmap: React.FC<{
  corr: CorrelationMatrixResult;
}> = ({ corr }) => {
  const { columns, matrix } = corr;
  if (columns.length === 0) {
    return <p className="text-sm text-slate-500">Sin suficientes columnas numéricas para matriz de correlación.</p>;
  }

  const getCellBg = (val: number) => {
    const abs = Math.min(1, Math.abs(val));
    if (val >= 0) {
      return `rgba(13, 148, 136, ${0.1 + abs * 0.78})`;
    }
    return `rgba(220, 38, 38, ${0.1 + abs * 0.65})`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            <th className="p-2 text-left font-medium text-slate-500 border-b border-slate-200">Variable</th>
            {columns.map((c) => (
              <th
                key={c}
                className="p-2 text-center font-medium text-slate-700 border-b border-slate-200 max-w-[110px] truncate"
                title={c}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {columns.map((rowCol, i) => (
            <tr key={rowCol}>
              <td className="p-2 font-medium text-slate-800 border-b border-slate-100 whitespace-nowrap">
                {rowCol}
              </td>
              {columns.map((colName, j) => {
                const v = matrix[i]?.[j] ?? 0;
                const darkText = Math.abs(v) < 0.55;
                return (
                  <td
                    key={colName}
                    className="p-2 text-center font-mono tabular-nums border border-white"
                    style={{
                      backgroundColor: getCellBg(v),
                      color: darkText ? "#0f172a" : "#ffffff",
                    }}
                    title={`${rowCol} vs ${colName}: ${v.toFixed(3)}`}
                  >
                    {v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export const CoefficientForestPlot: React.FC<{
  coefficients: CoefficientStat[];
}> = ({ coefficients }) => {
  const rows = coefficients.filter((c) => !c.isIntercept);
  if (rows.length === 0) return null;

  const vals: number[] = [0];
  for (const r of rows) {
    vals.push(r.estimate);
    if (r.ciLow !== null) vals.push(r.ciLow);
    if (r.ciHigh !== null) vals.push(r.ciHigh);
  }
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const pad = (maxV - minV) * 0.12 || 1;
  const lo = minV - pad;
  const hi = maxV + pad;
  const toX = (v: number) => 160 + ((v - lo) / (hi - lo)) * 310;
  const zeroX = toX(0);
  const svgH = rows.length * 34 + 32;

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 500 ${svgH}`} className="w-full min-w-[440px]">
        <line x1={zeroX} y1={10} x2={zeroX} y2={svgH - 22} stroke="#94a3b8" strokeDasharray="3 3" strokeWidth={1.2} />
        {rows.map((r, i) => {
          const y = 24 + i * 34;
          const xEst = toX(r.estimate);
          const xLo = r.ciLow !== null ? toX(r.ciLow) : xEst;
          const xHi = r.ciHigh !== null ? toX(r.ciHigh) : xEst;
          const color = r.estimate >= 0 ? "#0D9488" : "#DC2626";
          return (
            <g key={r.term}>
              <text x={8} y={y + 4} fontSize={10.5} fill="#1e293b" fontWeight={500}>
                {r.term.length > 22 ? r.term.slice(0, 21) + "…" : r.term}
              </text>
              {r.ciLow !== null && r.ciHigh !== null && (
                <>
                  <line x1={xLo} y1={y} x2={xHi} y2={y} stroke={color} strokeWidth={2} />
                  <line x1={xLo} y1={y - 4} x2={xLo} y2={y + 4} stroke={color} strokeWidth={1.5} />
                  <line x1={xHi} y1={y - 4} x2={xHi} y2={y + 4} stroke={color} strokeWidth={1.5} />
                </>
              )}
              <circle cx={xEst} cy={y} r={4.5} fill={color} stroke="#ffffff" strokeWidth={1.2} />
            </g>
          );
        })}
        <text x={zeroX} y={svgH - 6} fontSize={9.5} fill="#64748b" textAnchor="middle" fontFamily="JetBrains Mono">
          0 (Efecto nulo)
        </text>
      </svg>
    </div>
  );
};

export const ResidualsScatterChart: React.FC<{
  data: Array<{ fitted: number; residual: number }>;
}> = ({ data }) => (
  <div style={{ width: "100%", height: 230 }}>
    <ResponsiveContainer>
      <ScatterChart margin={{ top: 10, right: 16, left: 4, bottom: 12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" dataKey="fitted" name="Valor Ajustado" tick={{ fontSize: 10, fill: "#475569" }} />
        <YAxis type="number" dataKey="residual" name="Residuo" tick={{ fontSize: 10, fill: "#475569" }} width={52} />
        <ReferenceLine y={0} stroke="#0B1F3A" strokeDasharray="4 4" />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0B1F3A",
            border: "none",
            borderRadius: "8px",
            color: "#fff",
            fontSize: "12px",
          }}
        />
        <Scatter name="Residuos" data={data} fill="#0D9488" fillOpacity={0.75} />
      </ScatterChart>
    </ResponsiveContainer>
  </div>
);

export const QQPlotChart: React.FC<{
  data: Array<{ theoretical: number; sample: number }>;
}> = ({ data }) => (
  <div style={{ width: "100%", height: 230 }}>
    <ResponsiveContainer>
      <ComposedChart data={data} margin={{ top: 10, right: 16, left: 4, bottom: 12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" dataKey="theoretical" name="Cuantil Teórico N(0,1)" tick={{ fontSize: 10, fill: "#475569" }} />
        <YAxis type="number" dataKey="sample" name="Residuo Est." tick={{ fontSize: 10, fill: "#475569" }} width={44} />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0B1F3A",
            border: "none",
            borderRadius: "8px",
            color: "#fff",
            fontSize: "12px",
          }}
        />
        <Scatter name="Cuantiles" data={data} fill="#0B1F3A" fillOpacity={0.75} />
        <Line
          data={[
            { theoretical: -2.5, ref: -2.5 },
            { theoretical: 2.5, ref: 2.5 },
          ]}
          dataKey="ref"
          stroke="#0D9488"
          strokeWidth={1.5}
          dot={false}
          name="Normalidad 45°"
        />
      </ComposedChart>
    </ResponsiveContainer>
  </div>
);

export const ScenarioComparisonChart: React.FC<{
  scenarios: ScenarioResult[];
  targetUnit: string;
}> = ({ scenarios }) => {
  const data = scenarios.map((s) => ({
    name: s.name.replace("Escenario ", ""),
    y: Number(s.predictedY.toFixed(2)),
    type: s.type,
  }));

  return (
    <div style={{ width: "100%", height: 250 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 12, right: 16, left: 8, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#334155" }} />
          <YAxis tick={{ fontSize: 11, fill: "#475569" }} width={64} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0B1F3A",
              border: "none",
              borderRadius: "8px",
              color: "#fff",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="y" name="Resultado Estimado (Ŷ)" radius={[4, 4, 0, 0]}>
            {data.map((entry, idx) => {
              const fill =
                entry.type === "optimista"
                  ? "#0D9488"
                  : entry.type === "adverso"
                  ? "#DC2626"
                  : entry.type === "personalizado"
                  ? "#0284C7"
                  : entry.type === "probabilistico"
                  ? "#0F766E"
                  : "#0B1F3A";
              return <Cell key={idx} fill={fill} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const SensitivityTornadoChart: React.FC<{
  bars: SensitivityBar[];
}> = ({ bars }) => {
  const data = bars.map((b) => ({
    column: b.column,
    alza: Number(b.impactPositive.toFixed(2)),
    baja: Number(b.impactNegative.toFixed(2)),
  }));

  return (
    <div style={{ width: "100%", height: 240 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 20, left: 30, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: "#475569" }} />
          <YAxis type="category" dataKey="column" tick={{ fontSize: 11, fill: "#1e293b" }} width={115} />
          <ReferenceLine x={0} stroke="#0B1F3A" />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0B1F3A",
              border: "none",
              borderRadius: "8px",
              color: "#fff",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="alza" name="Desplazamiento +0.5σ" fill="#0D9488" radius={[0, 3, 3, 0]} />
          <Bar dataKey="baja" name="Desplazamiento -0.5σ" fill="#64748b" radius={[3, 0, 0, 3]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
