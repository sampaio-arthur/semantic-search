import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';
import { api, EvaluationQuery } from '@/lib/api';
const DEFAULT_DATASET_ID = 'beir/trec-covid';

export default function EvaluationQueries() {
  const { user, isLoading: authLoading } = useAuth();
  const navigate = useNavigate();

  const [queries, setQueries] = useState<EvaluationQuery[]>([]);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!authLoading && !user) navigate('/auth');
  }, [authLoading, user, navigate]);

  useEffect(() => {
    if (!user) return;
    void loadQueries();
  }, [user]);

  const loadQueries = async () => {
    setIsBusy(true);
    setError('');
    try {
      const data = await api.getEvaluationQueries(DEFAULT_DATASET_ID);
      setQueries(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro ao carregar queries');
    } finally {
      setIsBusy(false);
    }
  };

  const handleUseQuery = (query: string) => {
    navigate(`/chat?q=${encodeURIComponent(query)}`);
  };

  const withAnswer = queries.filter((q) => q.ideal_answer).length;

  if (authLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="typing-indicator">
          <span />
          <span />
          <span />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="h-14 flex items-center px-6 border-b border-border gap-4">
        <Button variant="ghost" size="sm" onClick={() => navigate('/chat')}>
          ← Voltar
        </Button>
        <h1 className="text-lg font-medium">Queries de Avaliação</h1>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Queries do gabarito experimental. Clique em "Usar query" para preencher o campo de busca automaticamente.
          </p>
          {queries.length > 0 && (
            <p className="text-xs text-muted-foreground shrink-0 ml-4">
              {withAnswer}/{queries.length} com gabarito
            </p>
          )}
        </div>

        {isBusy && (
          <p className="text-sm text-muted-foreground">Carregando...</p>
        )}

        {error && (
          <p className="text-sm text-destructive mb-4">{error}</p>
        )}

        {!isBusy && !error && queries.length === 0 && (
          <p className="text-sm text-muted-foreground">
            Nenhuma query de avaliação encontrada. Indexe um dataset primeiro.
          </p>
        )}

        {queries.length > 0 && (
          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left px-4 py-3 text-muted-foreground font-medium w-32">#</th>
                  <th className="text-left px-4 py-3 text-muted-foreground font-medium">Query / Gabarito</th>
                  <th className="px-4 py-3 w-40" />
                </tr>
              </thead>
              <tbody>
                {queries.map((q, idx) => (
                  <tr
                    key={q.query_id}
                    className={idx % 2 === 0 ? 'bg-background' : 'bg-muted/20'}
                  >
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground align-top">
                      {idx + 1}
                    </td>
                    <td className="px-4 py-3 align-top space-y-2">
                      <p>{q.query}</p>

                      {q.ideal_answer && (
                        <p className="text-xs text-green-400 line-clamp-2">
                          ✓ {q.ideal_answer}
                        </p>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right align-top">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleUseQuery(q.query)}
                      >
                        Usar query
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}
