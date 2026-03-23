import { Clock, FileText } from 'lucide-react';
import { SearchResponse } from '@/lib/api';

interface PipelinePanelProps {
  response: SearchResponse | null;
}

export function PipelinePanel({ response }: PipelinePanelProps) {
  if (!response?.comparison) return null;

  const pipelines = [
    {
      name: 'Clássico',
      color: 'text-blue-400',
      border: 'border-blue-400/30',
      count: response.comparison.classical.results?.length ?? 0,
      latency: response.comparison.classical.metrics?.total_time_ms,
      encode: response.comparison.classical.metrics?.encode_time_ms,
      search: response.comparison.classical.metrics?.search_time_ms,
    },
    {
      name: 'Quântico',
      color: 'text-purple-400',
      border: 'border-purple-400/30',
      count: response.comparison.quantum.results?.length ?? 0,
      latency: response.comparison.quantum.metrics?.total_time_ms,
      encode: response.comparison.quantum.metrics?.encode_time_ms,
      search: response.comparison.quantum.metrics?.search_time_ms,
    },
    ...(response.comparison.statistical
      ? [
          {
            name: 'Estatístico',
            color: 'text-amber-400',
            border: 'border-amber-400/30',
            count: response.comparison.statistical.results?.length ?? 0,
            latency: response.comparison.statistical.metrics?.total_time_ms,
            encode: response.comparison.statistical.metrics?.encode_time_ms,
            search: response.comparison.statistical.metrics?.search_time_ms,
          },
        ]
      : []),
  ];

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-4">
      <div className="rounded-2xl border border-border bg-background/80 p-4">
        <p className="text-sm font-semibold mb-3">Recuperação por Pipeline</p>
        <div className="grid grid-cols-3 gap-3">
          {pipelines.map((p) => (
            <div
              key={p.name}
              className={`rounded-lg border ${p.border} bg-card px-3 py-3 space-y-2`}
            >
              <p className={`text-xs font-semibold ${p.color}`}>{p.name}</p>
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <FileText className="h-3 w-3" />
                <span>{p.count} docs</span>
              </div>
              {p.latency !== undefined && (
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  <span>{p.latency.toFixed(1)} ms</span>
                </div>
              )}
              {(p.encode !== undefined || p.search !== undefined) && (
                <div className="text-[10px] text-muted-foreground/70 pl-[18px]">
                  {p.encode !== undefined && <span>enc {p.encode.toFixed(1)}</span>}
                  {p.encode !== undefined && p.search !== undefined && <span> · </span>}
                  {p.search !== undefined && <span>busca {p.search.toFixed(1)}</span>}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
