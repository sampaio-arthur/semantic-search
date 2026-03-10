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

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState('');
  const [savingId, setSavingId] = useState<string | null>(null);
  const [saveError, setSaveError] = useState('');

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

  const startEditing = (q: EvaluationQuery) => {
    setEditingId(q.query_id);
    setEditDraft(q.ideal_answer ?? '');
    setSaveError('');
  };

  const cancelEditing = () => {
    setEditingId(null);
    setEditDraft('');
    setSaveError('');
  };

  const saveIdealAnswer = async (q: EvaluationQuery) => {
    if (!editDraft.trim()) return;
    setSavingId(q.query_id);
    setSaveError('');
    try {
      await api.upsertBenchmarkLabel({
        dataset_id: DEFAULT_DATASET_ID,
        query_text: q.query,
        ideal_answer: editDraft.trim(),
      });
      setEditingId(null);
      setEditDraft('');
      await loadQueries();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Erro ao salvar');
    } finally {
      setSavingId(null);
    }
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

        {saveError && (
          <p className="text-sm text-destructive">{saveError}</p>
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
                  <th className="text-left px-4 py-3 text-muted-foreground font-medium w-32">Query ID</th>
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
                      {q.query_id}
                    </td>
                    <td className="px-4 py-3 align-top space-y-2">
                      <p>{q.query}</p>

                      {/* gabarito existente */}
                      {q.ideal_answer && editingId !== q.query_id && (
                        <p className="text-xs text-green-400 line-clamp-2">
                          ✓ {q.ideal_answer}
                        </p>
                      )}

                      {/* editor inline */}
                      {editingId === q.query_id && (
                        <div className="space-y-2 pt-1">
                          <textarea
                            autoFocus
                            className="w-full bg-background border border-border rounded-md px-3 py-2 text-sm min-h-20 resize-y"
                            placeholder="Descreva a resposta esperada..."
                            value={editDraft}
                            onChange={(e) => setEditDraft(e.target.value)}
                          />
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              onClick={() => saveIdealAnswer(q)}
                              disabled={savingId === q.query_id || !editDraft.trim()}
                            >
                              {savingId === q.query_id ? 'Salvando...' : 'Salvar'}
                            </Button>
                            <Button size="sm" variant="outline" onClick={cancelEditing} disabled={savingId === q.query_id}>
                              Cancelar
                            </Button>
                          </div>
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right align-top">
                      <div className="flex flex-col gap-1 items-end">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleUseQuery(q.query)}
                        >
                          Usar query
                        </Button>
                        {editingId !== q.query_id && (
                          <Button
                            size="sm"
                            variant={q.ideal_answer ? 'ghost' : 'outline'}
                            onClick={() => startEditing(q)}
                            className={q.ideal_answer ? 'text-muted-foreground text-xs' : 'text-xs'}
                          >
                            {q.ideal_answer ? 'Editar gabarito' : 'Atribuir gabarito'}
                          </Button>
                        )}
                      </div>
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
