import { SearchAlgorithmDetails, SearchResponse } from '@/lib/api';

interface ComparisonPanelProps {
  response: SearchResponse | null;
}

function metricNumber(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function metricPercent(value: number | null, maxValue: number | null): number {
  if (value === null || maxValue === null || maxValue <= 0) return 0;
  return Math.max(0, Math.min(100, (value / maxValue) * 100));
}

function MetricCell({
  value,
  maxValue,
}: {
  value: number | null;
  maxValue: number | null;
}) {
  if (value === null) return <span>-</span>;

  const percent = metricPercent(value, maxValue);

  return (
    <div className='min-w-[100px]'>
      <div className='text-foreground'>{value.toFixed(3)}</div>
      <div className='mt-1 h-1.5 w-full rounded-full bg-muted'>
        <div
          className='h-1.5 rounded-full bg-emerald-500'
          style={{ width: `${percent.toFixed(1)}%` }}
        />
      </div>
      <div className='mt-1 text-[10px] text-muted-foreground'>{percent.toFixed(0)}%</div>
    </div>
  );
}

function StepByStep({ title, details }: { title: string; details?: SearchAlgorithmDetails }) {
  const rawSteps = details?.debug?.steps;
  const steps = Array.isArray(rawSteps) ? rawSteps.filter((x): x is string => typeof x === 'string') : [];

  return (
    <div className='rounded-xl border border-border bg-card p-4 space-y-2'>
      <p className='text-sm font-medium'>{title}</p>
      {steps.length ? (
        <ol className='space-y-1 text-xs text-muted-foreground'>
          {steps.map((step, idx) => (
            <li key={`${title}-${idx}`}>
              <span className='text-foreground'>{idx + 1}.</span> {step}
            </li>
          ))}
        </ol>
      ) : (
        <p className='text-xs text-muted-foreground'>Passo a passo não disponível.</p>
      )}
    </div>
  );
}

export function ComparisonPanel({ response }: ComparisonPanelProps) {
  if (!response) return null;

  const comparison = response.comparison;
  const showComparison = Boolean(comparison);
  const classicalResults = comparison?.classical.results ?? response.results ?? [];
  const quantumResults = comparison?.quantum.results ?? [];
  const statisticalResults = comparison?.statistical?.results ?? [];
  const classicalMetrics = comparison?.classical.metrics;
  const quantumMetrics = comparison?.quantum.metrics;
  const statisticalMetrics = comparison?.statistical?.metrics;
  const hasStatistical = Boolean(comparison?.statistical);
  const hasIrLabels = Boolean(
    classicalMetrics?.has_labels || quantumMetrics?.has_labels || statisticalMetrics?.has_labels
  );
  const hasIdealAnswer = Boolean(
    classicalMetrics?.has_ideal_answer || quantumMetrics?.has_ideal_answer || statisticalMetrics?.has_ideal_answer
  );
  const irObservation = hasIrLabels
    ? 'Calculado com gabarito (qrels).'
    : 'Requer gabarito (qrels) para calcular.';
  const simObservation = hasIdealAnswer
    ? 'Similaridade semântica entre resposta recuperada (top-3 docs) e resposta ideal.'
    : 'Requer ideal_answer no gabarito.';
  const metricMax = 1;

  return (
    <div className='w-full max-w-4xl mx-auto px-4 pb-6'>
      <div className='rounded-2xl border border-border bg-background/80 p-4 space-y-4'>
        <div className='flex flex-wrap items-center justify-between gap-2'>
          <div>
            <p className='text-sm font-semibold'>Comparação de Busca</p>
            <p className='text-xs text-muted-foreground'>
              Objetivo: comparar três transformações vetoriais sobre a mesma base semântica (BERT).
              Variável experimental: método de transformação do embedding.
            </p>
          </div>
          <span className='text-xs text-muted-foreground'>Modo: {response.mode}</span>
        </div>

        {showComparison ? (
          <div className={`grid gap-3 ${hasStatistical ? 'md:grid-cols-3' : 'md:grid-cols-2'}`}>
            <StepByStep title='Passo a passo: Clássico' details={comparison?.classical.algorithm_details} />
            <StepByStep title='Passo a passo: Quântico' details={comparison?.quantum.algorithm_details} />
            {hasStatistical && (
              <StepByStep title='Passo a passo: Estatístico' details={comparison?.statistical?.algorithm_details} />
            )}
          </div>
        ) : (
          <StepByStep title='Passo a passo' details={response.algorithm_details} />
        )}

        {showComparison && (
          <div className='rounded-xl border border-border bg-card p-4'>
            <p className='text-sm font-medium mb-3'>Tabela Comparativa (efetividade + custo)</p>
            <div className='overflow-x-auto'>
              <table className='w-full text-xs'>
                <thead>
                  <tr className='text-left text-muted-foreground border-b border-border'>
                    <th className='py-2 pr-3'>Métrica</th>
                    <th className='py-2 pr-3'>Clássico</th>
                    <th className='py-2 pr-3'>Quântico</th>
                    {hasStatistical && <th className='py-2 pr-3'>Estatístico</th>}
                    <th className='py-2'>Observação</th>
                  </tr>
                </thead>
                <tbody className='text-muted-foreground'>
                  <tr className='border-b border-border/60'>
                    <td className='py-2 pr-3 text-foreground'>Total (ms)</td>
                    <td className='py-2 pr-3'>{classicalMetrics?.total_time_ms?.toFixed(1) ?? '-'}</td>
                    <td className='py-2 pr-3'>{quantumMetrics?.total_time_ms?.toFixed(1) ?? '-'}</td>
                    {hasStatistical && <td className='py-2 pr-3'>{statisticalMetrics?.total_time_ms?.toFixed(1) ?? '-'}</td>}
                    <td className='py-2'>Comparativo de custo/tempo (não é qualidade)</td>
                  </tr>
                  <tr className='border-b border-border/60'>
                    <td className='py-2 pr-3 text-foreground'>Docs recuperados</td>
                    <td className='py-2 pr-3'>{classicalResults.length}</td>
                    <td className='py-2 pr-3'>{quantumResults.length}</td>
                    {hasStatistical && <td className='py-2 pr-3'>{statisticalResults.length}</td>}
                    <td className='py-2'>Top-k retornado por pipeline</td>
                  </tr>
                  <tr className='border-b border-border/60'>
                    <td className='py-2 pr-3 text-foreground'>Precision@k</td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber(classicalMetrics?.precision_at_k)} maxValue={metricMax} />
                    </td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber(quantumMetrics?.precision_at_k)} maxValue={metricMax} />
                    </td>
                    {hasStatistical && (
                      <td className='py-2 pr-3'>
                        <MetricCell value={metricNumber(statisticalMetrics?.precision_at_k)} maxValue={metricMax} />
                      </td>
                    )}
                    <td className='py-2'>{irObservation}</td>
                  </tr>
                  <tr className='border-b border-border/60'>
                    <td className='py-2 pr-3 text-foreground'>Recall@k</td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber(classicalMetrics?.recall_at_k)} maxValue={metricMax} />
                    </td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber(quantumMetrics?.recall_at_k)} maxValue={metricMax} />
                    </td>
                    {hasStatistical && (
                      <td className='py-2 pr-3'>
                        <MetricCell value={metricNumber(statisticalMetrics?.recall_at_k)} maxValue={metricMax} />
                      </td>
                    )}
                    <td className='py-2'>{irObservation}</td>
                  </tr>
                  <tr className='border-b border-border/60'>
                    <td className='py-2 pr-3 text-foreground'>NDCG@k</td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber(classicalMetrics?.ndcg_at_k)} maxValue={metricMax} />
                    </td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber(quantumMetrics?.ndcg_at_k)} maxValue={metricMax} />
                    </td>
                    {hasStatistical && (
                      <td className='py-2 pr-3'>
                        <MetricCell value={metricNumber(statisticalMetrics?.ndcg_at_k)} maxValue={metricMax} />
                      </td>
                    )}
                    <td className='py-2'>{irObservation}</td>
                  </tr>
                  <tr className='border-b border-border/60'>
                    <td className='py-2 pr-3 text-foreground'>MRR@k</td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber((classicalMetrics as Record<string, unknown>)?.mrr as number | null)} maxValue={metricMax} />
                    </td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber((quantumMetrics as Record<string, unknown>)?.mrr as number | null)} maxValue={metricMax} />
                    </td>
                    {hasStatistical && (
                      <td className='py-2 pr-3'>
                        <MetricCell value={metricNumber((statisticalMetrics as Record<string, unknown>)?.mrr as number | null)} maxValue={metricMax} />
                      </td>
                    )}
                    <td className='py-2'>{irObservation}</td>
                  </tr>
                  <tr>
                    <td className='py-2 pr-3 text-foreground'>Answer Similarity</td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber(classicalMetrics?.answer_similarity)} maxValue={metricMax} />
                    </td>
                    <td className='py-2 pr-3'>
                      <MetricCell value={metricNumber(quantumMetrics?.answer_similarity)} maxValue={metricMax} />
                    </td>
                    {hasStatistical && (
                      <td className='py-2 pr-3'>
                        <MetricCell value={metricNumber(statisticalMetrics?.answer_similarity)} maxValue={metricMax} />
                      </td>
                    )}
                    <td className='py-2'>{simObservation}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        <p className='text-xs text-muted-foreground'>
          {hasIrLabels
            ? `Métricas IR reais exibidas para esta query com gabarito (qrels) encontrado.${hasIdealAnswer ? ' Answer Similarity calculada com ideal_answer do gabarito.' : ''}`
            : 'Sem gabarito correspondente para esta query no chat. Para métricas IR canônicas, use queries do BEIR (query_id) ou avaliação batch.'}
        </p>
      </div>
    </div>
  );
}
