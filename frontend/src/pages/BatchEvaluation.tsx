import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useAuth } from '@/contexts/AuthContext';
import {
  api,
  BatchEvaluationResult,
  BatchEvaluationStatus,
  BatchPipelineResult,
  BatchPerQueryResult,
} from '@/lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const DEFAULT_DATASET_ID = 'beir/trec-covid';
const CACHE_KEY = `qs:batchEval:v1:${DEFAULT_DATASET_ID}`;

const PIPELINE_COLORS: Record<string, string> = {
  classical: '#3B82F6',
  quantum: '#8B5CF6',
  statistical: '#10B981',
};

const METRIC_OPTIONS = [
  { key: 'ndcg_at_k', label: 'nDCG@25' },
  { key: 'recall_at_k', label: 'Recall@25' },
  { key: 'mrr', label: 'MRR@25' },
  { key: 'precision_at_k', label: 'Precision@25' },
  { key: 'answer_similarity', label: 'Answer Similarity' },
] as const;

type MetricKey = (typeof METRIC_OPTIONS)[number]['key'];

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleString('pt-BR');
}

export default function BatchEvaluation() {
  const { user, isLoading: authLoading } = useAuth();
  const navigate = useNavigate();

  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');
  const [progress, setProgress] = useState({ current_query: 0, total_queries: 0, current_pipeline: '', completed_pipelines: [] as string[] });
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<BatchEvaluationResult | null>(null);
  const [error, setError] = useState('');
  const [cachedAt, setCachedAt] = useState<number | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>('ndcg_at_k');
  const [sortField, setSortField] = useState<string>('query_text');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  useEffect(() => {
    if (!authLoading && !user) navigate('/auth');
  }, [authLoading, user, navigate]);

  // Load cached result on mount
  useEffect(() => {
    const raw = localStorage.getItem(CACHE_KEY);
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        setResult(parsed.result);
        setCachedAt(parsed.timestamp);
        setStatus('completed');
      } catch { /* ignore */ }
    }
  }, []);

  // Polling
  useEffect(() => {
    if (status !== 'running') return;
    const interval = setInterval(async () => {
      try {
        const res = await api.getBatchEvaluationStatus();
        if (res.status === 'completed' && res.result) {
          setResult(res.result);
          setStatus('completed');
          const ts = Date.now();
          setCachedAt(ts);
          localStorage.setItem(CACHE_KEY, JSON.stringify({ result: res.result, timestamp: ts }));
          clearInterval(interval);
        } else if (res.status === 'failed') {
          setError(res.error || 'Erro desconhecido');
          setStatus('failed');
          clearInterval(interval);
        } else {
          setProgress(res.progress);
          setElapsed(res.elapsed_seconds);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Erro de conexão');
        setStatus('failed');
        clearInterval(interval);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [status]);

  const handleStart = useCallback(async () => {
    setError('');
    setStatus('running');
    setProgress({ current_query: 0, total_queries: 0, current_pipeline: '', completed_pipelines: [] });
    try {
      await api.startBatchEvaluation(DEFAULT_DATASET_ID);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao iniciar');
      setStatus('failed');
    }
  }, []);

  const handleRerun = useCallback(() => {
    localStorage.removeItem(CACHE_KEY);
    setResult(null);
    setCachedAt(null);
    handleStart();
  }, [handleStart]);

  if (authLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="typing-indicator"><span /><span /><span /></div>
      </div>
    );
  }

  const pipelines = result?.pipelines || [];
  const queryCount = pipelines[0]?.query_count || 0;

  // Chart data for aggregated metrics
  const metricsChartData = [
    { metric: 'nDCG@25', ...Object.fromEntries(pipelines.map(p => [p.pipeline, p.mean_ndcg_at_k])) },
    { metric: 'Recall@25', ...Object.fromEntries(pipelines.map(p => [p.pipeline, p.mean_recall_at_k])) },
    { metric: 'MRR@25', ...Object.fromEntries(pipelines.map(p => [p.pipeline, p.mean_mrr])) },
    { metric: 'P@25', ...Object.fromEntries(pipelines.map(p => [p.pipeline, p.mean_precision_at_k])) },
    { metric: 'Ans. Sim.', ...Object.fromEntries(pipelines.map(p => [p.pipeline, p.mean_answer_similarity ?? 0])) },
  ];

  // Latency chart data
  const latencyData = pipelines.map(p => ({
    pipeline: p.pipeline.charAt(0).toUpperCase() + p.pipeline.slice(1),
    encode_time: p.mean_encode_time_ms ?? 0,
    search_time: p.mean_search_time_ms ?? 0,
  }));

  // Table rows for means
  const metricsRows = [
    { label: 'nDCG@25', key: 'mean_ndcg_at_k' as const },
    { label: 'Recall@25', key: 'mean_recall_at_k' as const },
    { label: 'MRR@25', key: 'mean_mrr' as const },
    { label: 'Precision@25', key: 'mean_precision_at_k' as const },
    { label: 'Answer Similarity', key: 'mean_answer_similarity' as const },
  ];

  function getBest(key: string): string {
    let best = '';
    let bestVal = -1;
    for (const p of pipelines) {
      const val = (p as Record<string, unknown>)[key] as number | null;
      if (val !== null && val !== undefined && val > bestVal) {
        bestVal = val;
        best = p.pipeline;
      }
    }
    return best;
  }

  // Per-query data merged across pipelines
  type MergedQuery = { query_id: string; query_text: string } & Record<string, number | null>;
  const mergedQueries: MergedQuery[] = [];
  const queryMap = new Map<string, MergedQuery>();
  for (const p of pipelines) {
    for (const q of p.per_query) {
      if (!queryMap.has(q.query_id)) {
        const entry: MergedQuery = { query_id: q.query_id, query_text: q.query_text };
        queryMap.set(q.query_id, entry);
        mergedQueries.push(entry);
      }
      const entry = queryMap.get(q.query_id)!;
      entry[`${p.pipeline}_ndcg_at_k`] = q.ndcg_at_k;
      entry[`${p.pipeline}_recall_at_k`] = q.recall_at_k;
      entry[`${p.pipeline}_mrr`] = q.mrr;
      entry[`${p.pipeline}_precision_at_k`] = q.precision_at_k;
      entry[`${p.pipeline}_answer_similarity`] = q.answer_similarity;
    }
  }

  // Sort merged queries
  const sortedQueries = [...mergedQueries].sort((a, b) => {
    const aVal = a[sortField] ?? '';
    const bVal = b[sortField] ?? '';
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDir === 'asc' ? aVal - bVal : bVal - aVal;
    }
    return sortDir === 'asc'
      ? String(aVal).localeCompare(String(bVal))
      : String(bVal).localeCompare(String(aVal));
  });

  // Per-query chart data
  const perQueryChartData = mergedQueries.map(q => ({
    query: q.query_text.length > 45 ? q.query_text.substring(0, 45) + '...' : q.query_text,
    classical: q[`classical_${selectedMetric}`] ?? 0,
    quantum: q[`quantum_${selectedMetric}`] ?? 0,
    statistical: q[`statistical_${selectedMetric}`] ?? 0,
  }));

  const progressPercent = progress.total_queries > 0
    ? Math.round(((progress.completed_pipelines.length * progress.total_queries + progress.current_query) / (3 * progress.total_queries)) * 100)
    : 0;

  function toggleSort(field: string) {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('desc');
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="h-14 flex items-center px-6 border-b border-border gap-4">
        <Button variant="ghost" size="sm" onClick={() => navigate('/chat')}>
          ← Voltar
        </Button>
        <h1 className="text-lg font-medium">Avaliação Batch Comparativa</h1>
        <span className="text-xs text-muted-foreground ml-auto">
          Dataset: {DEFAULT_DATASET_ID} | Queries: {queryCount || '—'} | Top-25
        </span>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        {/* Action buttons */}
        <div className="flex items-center gap-3">
          {status === 'idle' && !result && (
            <Button onClick={handleStart}>Executar Avaliação</Button>
          )}
          {(status === 'completed' || status === 'failed') && (
            <Button onClick={handleRerun} variant="outline">Reexecutar</Button>
          )}
          {cachedAt && (
            <span className="text-xs text-muted-foreground">
              Última execução: {formatTimestamp(cachedAt)}
            </span>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-destructive bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Progress */}
        {status === 'running' && (
          <div className="rounded-lg border border-border bg-card px-4 py-3 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium">
                Pipeline: {progress.current_pipeline || '—'} | Query {progress.current_query}/{progress.total_queries}
              </span>
              <span className="text-muted-foreground">{elapsed.toFixed(0)}s</span>
            </div>
            <Progress value={progressPercent} />
            {progress.completed_pipelines.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Concluídos: {progress.completed_pipelines.join(', ')}
              </p>
            )}
          </div>
        )}

        {/* Results */}
        {result && pipelines.length > 0 && (
          <>
            {/* Grouped bar chart - metrics */}
            <div className="rounded-xl border border-border bg-card p-4">
              <p className="text-sm font-semibold mb-4">Médias por Pipeline</p>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={metricsChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v: number) => v.toFixed(4)} />
                  <Legend />
                  <Bar dataKey="classical" fill={PIPELINE_COLORS.classical} name="Classical" />
                  <Bar dataKey="quantum" fill={PIPELINE_COLORS.quantum} name="Quantum" />
                  <Bar dataKey="statistical" fill={PIPELINE_COLORS.statistical} name="Statistical" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Means table */}
            <div className="rounded-xl border border-border bg-card p-4">
              <p className="text-sm font-semibold mb-3">Tabela de Médias</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-muted-foreground border-b border-border">
                      <th className="py-2 pr-3">Métrica</th>
                      <th className="py-2 pr-3">Classical</th>
                      <th className="py-2 pr-3">Quantum</th>
                      <th className="py-2 pr-3">Statistical</th>
                      <th className="py-2">Melhor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metricsRows.map(row => {
                      const best = getBest(row.key);
                      return (
                        <tr key={row.key} className="border-b border-border/60">
                          <td className="py-2 pr-3 text-foreground font-medium">{row.label}</td>
                          {pipelines.map(p => {
                            const val = (p as Record<string, unknown>)[row.key] as number | null;
                            const isBest = p.pipeline === best;
                            return (
                              <td key={p.pipeline} className={`py-2 pr-3 ${isBest ? 'font-bold text-emerald-500' : ''}`}>
                                {val !== null && val !== undefined ? val.toFixed(4) : '—'}
                              </td>
                            );
                          })}
                          <td className="py-2 text-emerald-500 font-medium capitalize">{best}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Latency chart */}
            <div className="rounded-xl border border-border bg-card p-4">
              <p className="text-sm font-semibold mb-4">Latência Média (ms)</p>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={latencyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="pipeline" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v: number) => `${v.toFixed(1)} ms`} />
                  <Legend />
                  <Bar dataKey="encode_time" stackId="a" fill="#93C5FD" name="Encode (ms)" />
                  <Bar dataKey="search_time" stackId="a" fill="#3B82F6" name="Search (ms)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Per-query metric selector + chart */}
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="flex items-center gap-3 mb-4">
                <p className="text-sm font-semibold">Detalhamento por Query</p>
                <select
                  value={selectedMetric}
                  onChange={e => setSelectedMetric(e.target.value as MetricKey)}
                  className="text-xs bg-background border border-border rounded px-2 py-1"
                >
                  {METRIC_OPTIONS.map(opt => (
                    <option key={opt.key} value={opt.key}>{opt.label}</option>
                  ))}
                </select>
              </div>
              <div style={{ overflowY: 'auto', maxHeight: 600 }}>
                <ResponsiveContainer width="100%" height={Math.max(400, perQueryChartData.length * 35)}>
                  <BarChart data={perQueryChartData} layout="vertical" margin={{ left: 200 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
                    <YAxis dataKey="query" type="category" width={190} tick={{ fontSize: 10 }} />
                    <Tooltip formatter={(v: number) => v.toFixed(4)} />
                    <Legend />
                    <Bar dataKey="classical" fill={PIPELINE_COLORS.classical} name="Classical" />
                    <Bar dataKey="quantum" fill={PIPELINE_COLORS.quantum} name="Quantum" />
                    <Bar dataKey="statistical" fill={PIPELINE_COLORS.statistical} name="Statistical" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Detailed per-query table */}
            <div className="rounded-xl border border-border bg-card p-4">
              <p className="text-sm font-semibold mb-3">Tabela Detalhada por Query</p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-left text-muted-foreground border-b border-border">
                      <th className="py-2 pr-2 cursor-pointer hover:text-foreground" onClick={() => toggleSort('query_text')}>
                        Query {sortField === 'query_text' ? (sortDir === 'asc' ? '↑' : '↓') : ''}
                      </th>
                      {['nDCG', 'Recall', 'MRR', 'P@25', 'Ans.Sim.'].map(metric => {
                        const metricKeys: Record<string, string> = {
                          'nDCG': 'ndcg_at_k',
                          'Recall': 'recall_at_k',
                          'MRR': 'mrr',
                          'P@25': 'precision_at_k',
                          'Ans.Sim.': 'answer_similarity',
                        };
                        const mk = metricKeys[metric];
                        return ['classical', 'quantum', 'statistical'].map(pl => {
                          const field = `${pl}_${mk}`;
                          const abbr = pl.charAt(0).toUpperCase();
                          return (
                            <th key={field} className="py-2 pr-1 cursor-pointer hover:text-foreground whitespace-nowrap" onClick={() => toggleSort(field)}>
                              {metric} ({abbr}){sortField === field ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                            </th>
                          );
                        });
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedQueries.map((q, idx) => (
                      <tr key={q.query_id} className={idx % 2 === 0 ? 'bg-background' : 'bg-muted/20'}>
                        <td className="py-1.5 pr-2 max-w-[200px] truncate" title={q.query_text}>{q.query_text}</td>
                        {['ndcg_at_k', 'recall_at_k', 'mrr', 'precision_at_k', 'answer_similarity'].flatMap(mk =>
                          ['classical', 'quantum', 'statistical'].map(pl => {
                            const val = q[`${pl}_${mk}`];
                            return (
                              <td key={`${pl}_${mk}`} className="py-1.5 pr-1 text-right tabular-nums">
                                {val !== null && val !== undefined ? (val as number).toFixed(4) : '—'}
                              </td>
                            );
                          })
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
