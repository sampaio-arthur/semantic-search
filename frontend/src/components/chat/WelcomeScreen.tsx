import { Sparkles, FlaskConical, ArrowRight } from 'lucide-react';

interface WelcomeScreenProps {
  onQueryClick?: (query: string) => void;
}

export function WelcomeScreen({ onQueryClick }: WelcomeScreenProps) {
  const exampleQueries = [
    'what is the origin of COVID-19',
    'how does the coronavirus respond to changes in the weather',
    'what causes death from Covid-19?',
    'what drugs have been active against SARS-CoV or SARS-CoV-2 in animal studies?',
  ];

  const pipelines = [
    {
      name: 'Clássico',
      description: 'SBERT → PCA(64) → L2',
      color: 'text-blue-400',
    },
    {
      name: 'Quântico',
      description: 'SBERT → PCA → Circuito Quântico → Hellinger → PCA(64) → L2',
      color: 'text-purple-400',
    },
    {
      name: 'Estatístico',
      description: 'SBERT → PCA(128) → TruncatedSVD(64) → L2',
      color: 'text-amber-400',
    },
  ];

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4">
      <div className="text-center fade-in max-w-3xl">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-success/20 to-success/5 mb-6">
          <Sparkles className="h-8 w-8 text-success" />
        </div>
        <h1 className="text-3xl font-semibold text-foreground mb-2">
          Comparação de Transformações Vetoriais
        </h1>
        <p className="text-muted-foreground max-w-xl mx-auto">
          Esta plataforma compara o desempenho de <strong className="text-foreground">três pipelines de transformação vetorial</strong> aplicados ao mesmo embedding base (Sentence-BERT) para recuperação de informação no dataset BEIR trec-covid.
        </p>

        <div className="mt-8 rounded-xl border border-border bg-card/50 p-5 text-left">
          <div className="flex items-center gap-2 mb-3">
            <FlaskConical className="h-4 w-4 text-success" />
            <p className="text-sm font-medium text-foreground">Os três pipelines</p>
          </div>
          <div className="grid gap-3">
            {pipelines.map((p) => (
              <div key={p.name} className="flex items-baseline gap-2">
                <span className={`text-sm font-semibold ${p.color} min-w-[90px]`}>{p.name}</span>
                <span className="text-xs text-muted-foreground font-mono">{p.description}</span>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-3 border-t border-border pt-3">
            Todos produzem vetores de 64 dimensões normalizados por L2. A única variável experimental é a transformação aplicada — todas as demais variáveis são controladas (modelo base, métrica de similaridade, métricas de avaliação, dataset).
          </p>
        </div>

        <div className="mt-4 rounded-xl border border-border bg-card/50 p-5 text-left">
          <p className="text-sm font-medium text-foreground mb-3">Experimente uma consulta</p>
          <div className="grid gap-2">
            {exampleQueries.map((query) => (
              <button
                key={query}
                type="button"
                onClick={() => onQueryClick?.(query)}
                className="group flex items-center justify-between rounded-md border border-border px-3 py-2 text-xs text-muted-foreground hover:border-success/50 hover:text-foreground hover:bg-card transition-colors text-left"
              >
                <span>{query}</span>
                <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 ml-2" />
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
