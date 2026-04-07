import { SearchResponse } from '@/lib/api';

interface ComparisonPanelProps {
  response: SearchResponse | null;
  topK?: number;
}

function metricNumber(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function MetricBar({ value }: { value: number | null }) {
  if (value === null) return <span className="text-muted-foreground">—</span>;

  const percent = Math.max(0, Math.min(100, value * 100));

  return (
    <div className="min-w-[80px]">
      <div className="text-foreground font-medium">{value.toFixed(3)}</div>
      <div className="mt-1 h-1.5 w-full rounded-full bg-muted">
        <div
          className="h-1.5 rounded-full bg-emerald-500 transition-all"
          style={{ width: `${percent.toFixed(1)}%` }}
        />
      </div>
    </div>
  );
}

interface MetricRowProps {
  label: string;
  classical: number | null;
  quantum: number | null;
  statistical: number | null;
  hasStatistical: boolean;
}

function MetricRow({ label, classical, quantum, statistical, hasStatistical }: MetricRowProps) {
  return (
    <tr className="border-b border-border/40 last:border-0">
      <td className="py-2.5 pr-3 text-foreground font-medium">{label}</td>
      <td className="py-2.5 pr-3">
        <MetricBar value={classical} />
      </td>
      <td className="py-2.5 pr-3">
        <MetricBar value={quantum} />
      </td>
      {hasStatistical && (
        <td className="py-2.5 pr-3">
          <MetricBar value={statistical} />
        </td>
      )}
    </tr>
  );
}

export function ComparisonPanel({ response, topK = 25 }: ComparisonPanelProps) {
  if (!response) return null;

  const comparison = response.comparison;
  if (!comparison) return null;

  const classicalMetrics = comparison.classical.metrics;
  const quantumMetrics = comparison.quantum.metrics;
  const statisticalMetrics = comparison.statistical?.metrics;
  const hasStatistical = Boolean(comparison.statistical);

  const hasIrLabels = Boolean(
    classicalMetrics?.has_labels || quantumMetrics?.has_labels || statisticalMetrics?.has_labels
  );

  const k = classicalMetrics?.k ?? topK;
  const metrics = [
    {
      label: `P@${k}`,
      c: metricNumber(classicalMetrics?.precision_at_k),
      q: metricNumber(quantumMetrics?.precision_at_k),
      s: metricNumber(statisticalMetrics?.precision_at_k),
    },
    {
      label: `Recall@${k}`,
      c: metricNumber(classicalMetrics?.recall_at_k),
      q: metricNumber(quantumMetrics?.recall_at_k),
      s: metricNumber(statisticalMetrics?.recall_at_k),
    },
    {
      label: `nDCG@${k}`,
      c: metricNumber(classicalMetrics?.ndcg_at_k),
      q: metricNumber(quantumMetrics?.ndcg_at_k),
      s: metricNumber(statisticalMetrics?.ndcg_at_k),
    },
    {
      label: `MRR@${k}`,
      c: metricNumber((classicalMetrics as Record<string, unknown>)?.mrr as number | null),
      q: metricNumber((quantumMetrics as Record<string, unknown>)?.mrr as number | null),
      s: metricNumber((statisticalMetrics as Record<string, unknown>)?.mrr as number | null),
    },
  ];

  const allNull = metrics.every((m) => m.c === null && m.q === null && m.s === null);

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-6">
      <div className="rounded-2xl border border-border bg-background/80 p-4 space-y-3">
        <div>
          <p className="text-sm font-semibold">Métricas de Qualidade</p>
          <p className="text-xs text-muted-foreground">
            {hasIrLabels
              ? 'Calculado com gabarito (qrels) do BEIR.'
              : 'Sem gabarito para esta query — use queries do BEIR para ver métricas IR.'}
          </p>
        </div>

        {!allNull && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground border-b border-border">
                  <th className="py-2 pr-3">Métrica</th>
                  <th className="py-2 pr-3 text-blue-400">Clássico</th>
                  <th className="py-2 pr-3 text-purple-400">Quântico</th>
                  {hasStatistical && (
                    <th className="py-2 pr-3 text-amber-400">Estatístico</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {metrics.map((m) => (
                  <MetricRow
                    key={m.label}
                    label={m.label}
                    classical={m.c}
                    quantum={m.q}
                    statistical={m.s}
                    hasStatistical={hasStatistical}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </div>
  );
}
