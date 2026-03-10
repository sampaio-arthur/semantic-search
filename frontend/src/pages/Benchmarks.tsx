import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';
import { api, BenchmarkLabel, EvaluationQuery } from '@/lib/api';

const DEFAULT_DATASET_ID = 'beir/trec-covid';

export default function Benchmarks() {
  const { user, isLoading: authLoading } = useAuth();
  const navigate = useNavigate();

  const [activeDatasetId] = useState(DEFAULT_DATASET_ID);
  const [labels, setLabels] = useState<BenchmarkLabel[]>([]);
  const [queries, setQueries] = useState<EvaluationQuery[]>([]);
  const [isBusy, setIsBusy] = useState(false);

  const [queryText, setQueryText] = useState('');
  const [idealAnswer, setIdealAnswer] = useState('');
  const [message, setMessage] = useState('');

  // inline editing state: query_id -> draft text
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState('');
  const [savingId, setSavingId] = useState<string | null>(null);

  // search/filter for the queries list
  const [queryFilter, setQueryFilter] = useState('');

  useEffect(() => {
    if (!authLoading && !user) navigate('/auth');
  }, [authLoading, user, navigate]);

  useEffect(() => {
    if (!user) return;
    void loadAll(activeDatasetId);
  }, [user]);

  const loadAll = async (datasetId: string) => {
    setIsBusy(true);
    try {
      const [savedLabels, evalQueries] = await Promise.all([
        api.listBenchmarkLabels(datasetId),
        api.getEvaluationQueries(datasetId),
      ]);
      setLabels(savedLabels);
      setQueries(evalQueries);
      setMessage('');
    } catch (error) {
      console.error(error);
      setMessage(error instanceof Error ? error.message : 'Erro ao carregar dados');
    } finally {
      setIsBusy(false);
    }
  };

  const handleSave = async () => {
    if (!queryText.trim() || !idealAnswer.trim()) {
      setMessage('Preencha pergunta e resposta ideal.');
      return;
    }

    setIsBusy(true);
    try {
      await api.upsertBenchmarkLabel({
        dataset_id: activeDatasetId,
        query_text: queryText.trim(),
        ideal_answer: idealAnswer.trim(),
      });
      await loadAll(activeDatasetId);
      setQueryText('');
      setIdealAnswer('');
      setMessage('Gabarito salvo com sucesso.');
    } catch (error) {
      console.error(error);
      setMessage(error instanceof Error ? error.message : 'Erro ao salvar gabarito');
    } finally {
      setIsBusy(false);
    }
  };

  const handleDelete = async (item: BenchmarkLabel) => {
    setIsBusy(true);
    try {
      await api.deleteBenchmarkLabel(item.dataset_id, item.benchmark_id);
      await loadAll(item.dataset_id);
      setMessage('Gabarito removido.');
    } catch (error) {
      console.error(error);
      setMessage(error instanceof Error ? error.message : 'Erro ao remover gabarito');
    } finally {
      setIsBusy(false);
    }
  };

  const startEditing = (q: EvaluationQuery) => {
    setEditingId(q.query_id);
    setEditDraft(q.ideal_answer ?? '');
  };

  const cancelEditing = () => {
    setEditingId(null);
    setEditDraft('');
  };

  const saveIdealAnswer = async (q: EvaluationQuery) => {
    if (!editDraft.trim()) return;
    setSavingId(q.query_id);
    try {
      await api.upsertBenchmarkLabel({
        dataset_id: activeDatasetId,
        query_text: q.query,
        ideal_answer: editDraft.trim(),
      });
      setEditingId(null);
      setEditDraft('');
      await loadAll(activeDatasetId);
    } catch (error) {
      console.error(error);
      setMessage(error instanceof Error ? error.message : 'Erro ao salvar resposta ideal');
    } finally {
      setSavingId(null);
    }
  };

  const filteredQueries = queries.filter((q) =>
    queryFilter === '' || q.query.toLowerCase().includes(queryFilter.toLowerCase())
  );

  const withAnswer = queries.filter((q) => q.ideal_answer).length;

  if (authLoading) {
    return <div className='min-h-screen bg-background' />;
  }

  return (
    <div className='min-h-screen bg-background text-foreground'>
      <div className='max-w-5xl mx-auto px-4 py-6 space-y-6'>
        <div className='flex items-center justify-between gap-3'>
          <div>
            <h1 className='text-2xl font-semibold'>Gabaritos de Acuracia</h1>
            <p className='text-sm text-muted-foreground'>
              Cadastre pergunta e resposta ideal. O sistema infere docs relevantes para medir ranking e precisao classico vs quantico.
            </p>
          </div>
          <Button variant='outline' onClick={() => navigate('/chat')}>Voltar ao Chat</Button>
        </div>

        {/* Nova query manual */}
        <div className='rounded-xl border border-border p-4 space-y-4'>
          <div className='space-y-1'>
            <p className='text-sm font-medium'>Dataset padrao</p>
            <p className='text-sm text-muted-foreground'>{activeDatasetId}</p>
          </div>

          <div className='space-y-2'>
            <label className='text-sm font-medium'>Pergunta a avaliar</label>
            <textarea
              className='w-full bg-background border border-border rounded-md px-3 py-2 text-sm min-h-24'
              placeholder='Ex: qual o impacto das nanoparticulas?'
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
            />
          </div>

          <div className='space-y-2'>
            <label className='text-sm font-medium'>Resposta ideal (gabarito)</label>
            <textarea
              className='w-full bg-background border border-border rounded-md px-3 py-2 text-sm min-h-32'
              placeholder='Descreva a resposta esperada para comparar a qualidade da resposta gerada.'
              value={idealAnswer}
              onChange={(e) => setIdealAnswer(e.target.value)}
            />
          </div>

          <div className='flex items-center gap-2'>
            <Button onClick={handleSave} disabled={isBusy}>Salvar gabarito</Button>
            <Button variant='outline' onClick={() => loadAll(activeDatasetId)} disabled={isBusy}>Atualizar lista</Button>
          </div>

          {message && <p className='text-sm text-muted-foreground'>{message}</p>}
        </div>

        {/* Queries existentes do dataset para atribuir ideal_answer */}
        <div className='rounded-xl border border-border p-4 space-y-3'>
          <div className='flex items-center justify-between gap-2'>
            <div>
              <h2 className='text-lg font-medium'>Queries do dataset</h2>
              <p className='text-xs text-muted-foreground'>
                {withAnswer} de {queries.length} queries com resposta ideal atribuida
              </p>
            </div>
          </div>

          <input
            type='text'
            className='w-full bg-background border border-border rounded-md px-3 py-2 text-sm'
            placeholder='Filtrar queries...'
            value={queryFilter}
            onChange={(e) => setQueryFilter(e.target.value)}
          />

          {filteredQueries.length === 0 ? (
            <p className='text-sm text-muted-foreground'>
              {queries.length === 0 ? 'Nenhuma query encontrada. Indexe um dataset primeiro.' : 'Nenhuma query corresponde ao filtro.'}
            </p>
          ) : (
            <div className='space-y-2 max-h-[600px] overflow-y-auto pr-1'>
              {filteredQueries.map((q) => (
                <div
                  key={q.query_id}
                  className={`rounded-lg border p-3 space-y-2 ${q.ideal_answer ? 'border-green-600/40 bg-green-950/10' : 'border-border'}`}
                >
                  <div className='flex items-start justify-between gap-2'>
                    <div className='flex-1 min-w-0'>
                      <p className='text-xs text-muted-foreground font-mono'>{q.query_id}</p>
                      <p className='text-sm mt-0.5'>{q.query}</p>
                    </div>
                    {editingId !== q.query_id && (
                      <Button
                        variant='outline'
                        size='sm'
                        onClick={() => startEditing(q)}
                        disabled={isBusy}
                        className='shrink-0'
                      >
                        {q.ideal_answer ? 'Editar' : 'Atribuir gabarito'}
                      </Button>
                    )}
                  </div>

                  {q.ideal_answer && editingId !== q.query_id && (
                    <div className='pl-0 space-y-0.5'>
                      <p className='text-xs font-medium text-green-400'>Resposta ideal</p>
                      <p className='text-xs text-muted-foreground whitespace-pre-wrap line-clamp-3'>{q.ideal_answer}</p>
                    </div>
                  )}

                  {editingId === q.query_id && (
                    <div className='space-y-2'>
                      <textarea
                        autoFocus
                        className='w-full bg-background border border-border rounded-md px-3 py-2 text-sm min-h-24'
                        placeholder='Descreva a resposta esperada...'
                        value={editDraft}
                        onChange={(e) => setEditDraft(e.target.value)}
                      />
                      <div className='flex gap-2'>
                        <Button
                          size='sm'
                          onClick={() => saveIdealAnswer(q)}
                          disabled={savingId === q.query_id || !editDraft.trim()}
                        >
                          {savingId === q.query_id ? 'Salvando...' : 'Salvar'}
                        </Button>
                        <Button size='sm' variant='outline' onClick={cancelEditing} disabled={savingId === q.query_id}>
                          Cancelar
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Gabaritos salvos (via BenchmarkLabel) */}
        <div className='rounded-xl border border-border p-4 space-y-3'>
          <h2 className='text-lg font-medium'>Gabaritos salvos</h2>
          {labels.length === 0 ? (
            <p className='text-sm text-muted-foreground'>Nenhum gabarito salvo para este dataset.</p>
          ) : (
            <div className='space-y-2'>
              {labels.map((item) => (
                <div key={item.dataset_id + ':' + item.benchmark_id} className='rounded-lg border border-border p-3'>
                  <div className='flex items-start justify-between gap-2'>
                    <div className='space-y-1'>
                      <p className='text-sm font-semibold'>Pergunta</p>
                      <p className='text-sm text-muted-foreground'>{item.query_text}</p>
                      <p className='text-sm font-semibold pt-2'>Resposta ideal</p>
                      <p className='text-sm text-muted-foreground whitespace-pre-wrap'>{item.ideal_answer}</p>
                      <p className='text-xs text-muted-foreground pt-1'>Docs relevantes inferidos: {item.relevant_doc_ids.length}</p>
                    </div>
                    <Button variant='destructive' size='sm' onClick={() => handleDelete(item)} disabled={isBusy}>Excluir</Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
