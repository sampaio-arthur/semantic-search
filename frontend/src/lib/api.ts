const ENV_API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;
if (import.meta.env.PROD && !ENV_API_BASE_URL) {
  throw new Error('VITE_API_BASE_URL must be configured in production builds');
}
const API_BASE_URL = ENV_API_BASE_URL || 'http://localhost:8000';
const REQUEST_TIMEOUT_MS = 60000;
const INDEX_POLL_INTERVAL_MS = 1500;

export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  refresh_token?: string;
}

export interface Conversation {
  id: number;
  title: string;
  created_at: string;
}

export interface Message {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface DatasetSummary {
  dataset_id: string;
  name: string;
  description: string;
  document_count: number;
  query_count: number;
}

export interface DatasetDetail {
  dataset_id: string;
  name: string;
  description: string;
  queries: { query_id: string; query: string; relevant_count: number }[];
  documents: { doc_id: string }[];
}

export interface BenchmarkLabelInput {
  dataset_id: string;
  query_text: string;
}

export interface BenchmarkLabel {
  benchmark_id: string;
  dataset_id: string;
  query_text: string;
  ideal_answer: string;
  relevant_doc_ids: string[];
}

export interface SearchResult {
  doc_id: string;
  text: string;
  score: number;
}

export interface SearchMetrics {
  accuracy_at_k?: number | null;
  precision_at_k?: number | null;
  recall_at_k?: number | null;
  f1_at_k?: number | null;
  mrr?: number | null;
  ndcg_at_k?: number | null;
  spearman?: number | null;
  answer_similarity?: number | null;
  has_ideal_answer: boolean;
  encode_time_ms: number;
  search_time_ms: number;
  total_time_ms: number;
  k: number;
  candidate_k: number;
  has_labels: boolean;
  debug?: Record<string, unknown> | null;
}

export interface SearchAlgorithmDetails {
  algorithm: string;
  comparator: string;
  candidate_strategy: string;
  description: string;
  debug?: Record<string, unknown> | null;
}

export interface SearchResponseLite {
  results: SearchResult[];
  answer?: string;
  metrics?: SearchMetrics;
  algorithm_details?: SearchAlgorithmDetails;
}

export interface SearchComparison {
  classical: SearchResponseLite;
  quantum: SearchResponseLite;
  statistical?: SearchResponseLite;
}

export interface SearchResponse {
  query: string;
  mode: string;
  results: SearchResult[];
  answer?: string;
  metrics?: SearchMetrics;
  comparison?: SearchComparison;
  comparison_metrics?: {
    top_k: number;
    common_doc_ids: string[];
    common_classical_quantum?: string[];
    common_classical_statistical?: string[];
    common_quantum_statistical?: string[];
  };
  algorithm_details?: SearchAlgorithmDetails;
}

export interface EvaluationQuery {
  query_id: string;
  query: string;
  ideal_answer: string | null;
}

export interface BatchEvaluationStatus {
  status: 'idle' | 'running' | 'completed' | 'failed';
  progress: {
    current_query: number;
    total_queries: number;
    current_pipeline: string;
    completed_pipelines: string[];
  };
  elapsed_seconds: number;
  result: BatchEvaluationResult | null;
  error: string | null;
}

export interface BatchEvaluationResult {
  dataset_id: string;
  k: number;
  pipelines: BatchPipelineResult[];
}

export interface BatchPipelineResult {
  pipeline: string;
  mean_precision_at_k: number;
  mean_recall_at_k: number;
  mean_ndcg_at_k: number;
  mean_mrr: number;
  mean_answer_similarity: number | null;
  mean_encode_time_ms: number | null;
  mean_search_time_ms: number | null;
  mean_total_time_ms: number | null;
  query_count: number;
  per_query: BatchPerQueryResult[];
}

export interface BatchPerQueryResult {
  query_id: string;
  query_text: string;
  precision_at_k: number;
  recall_at_k: number;
  ndcg_at_k: number;
  mrr: number;
  answer_similarity: number | null;
  encode_time_ms: number | null;
  search_time_ms: number | null;
  total_time_ms: number | null;
}

export interface DatasetIndexStatus {
  dataset_id: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  started_at?: string | null;
  finished_at?: string | null;
  updated_at?: string | null;
  indexed_count?: number;
  total_hint?: number | null;
  error?: string | null;
  result?: Record<string, unknown> | null;
  accepted?: boolean;
}

type DatasetIndexStatusListener = (status: DatasetIndexStatus) => void;

class ApiClient {
  private getToken(): string | null {
    return localStorage.getItem('access_token');
  }

  private getRefreshToken(): string | null {
    return localStorage.getItem('refresh_token');
  }

  private setTokens(accessToken: string, refreshToken?: string): void {
    localStorage.setItem('access_token', accessToken);
    if (refreshToken) {
      localStorage.setItem('refresh_token', refreshToken);
    }
  }

  private withAuthHeader(headers?: HeadersInit): Headers {
    const merged = new Headers(headers || {});
    const token = this.getToken();
    if (token) {
      merged.set('Authorization', `Bearer ${token}`);
    }
    return merged;
  }

  private async tryRefreshAccessToken(): Promise<boolean> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      return false;
    }
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!response.ok) {
      this.logout();
      return false;
    }
    const data = await response.json();
    if (!data?.access_token) {
      this.logout();
      return false;
    }
    this.setTokens(data.access_token, data.refresh_token);
    return true;
  }

  private async fetchWithAuth(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS): Promise<Response> {
    let response = await this.fetchWithTimeout(
      input,
      {
        ...init,
        headers: this.withAuthHeader(init.headers),
      },
      timeoutMs
    );
    if (response.status !== 401) {
      return response;
    }
    const refreshed = await this.tryRefreshAccessToken();
    if (!refreshed) {
      return response;
    }
    response = await this.fetchWithTimeout(
      input,
      {
        ...init,
        headers: this.withAuthHeader(init.headers),
      },
      timeoutMs
    );
    return response;
  }

  private getHeaders(json = true): HeadersInit {
    const headers: HeadersInit = {};
    const token = this.getToken();

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    if (json) {
      headers['Content-Type'] = 'application/json';
    }

    return headers;
  }

  private async fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

    try {
      return await fetch(input, { ...init, signal: controller.signal });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new Error('Tempo limite excedido ao aguardar resposta do servidor');
      }
      throw error;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  private async readErrorDetail(response: Response, fallback: string): Promise<string> {
    try {
      const data = await response.json();
      if (typeof data?.detail === 'string' && data.detail.trim()) {
        return data.detail;
      }
    } catch {
      // ignore json parsing errors and fallback
    }
    return fallback;
  }

  async register(email: string, password: string): Promise<User> {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Erro ao registrar');
    }

    return response.json();
  }

  async login(email: string, password: string): Promise<LoginResponse> {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Credenciais invalidas');
    }

    const data = await response.json();
    this.setTokens(data.access_token, data.refresh_token);
    return data;
  }

  async getMe(): Promise<User> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/auth/me`, {
      headers: this.getHeaders(false),
    });

    if (!response.ok) {
      throw new Error('Nao autenticado');
    }

    return response.json();
  }

  logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }

  async getConversations(): Promise<Conversation[]> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/conversations`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Erro ao carregar conversas');
    }

    return response.json();
  }

  async createConversation(title: string): Promise<Conversation> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/conversations`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ title }),
    });

    if (!response.ok) {
      throw new Error('Erro ao criar conversa');
    }

    return response.json();
  }

  async deleteConversation(id: number): Promise<void> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/conversations/${id}`, {
      method: 'DELETE',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Erro ao remover conversa');
    }
  }

  async getConversation(id: number): Promise<ConversationDetail> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/conversations/${id}`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Conversa nao encontrada');
    }

    return response.json();
  }

  async addMessage(conversationId: number, role: 'user' | 'assistant' | 'system', content: string): Promise<Message> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ role, content }),
    });

    if (!response.ok) {
      throw new Error('Erro ao enviar mensagem');
    }

    return response.json();
  }

  async startDatasetIndex(datasetId: string, forceReindex = false): Promise<DatasetIndexStatus> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/search/dataset/index`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ dataset_id: datasetId, force_reindex: forceReindex }),
    }, REQUEST_TIMEOUT_MS);

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao indexar dataset');
      throw new Error(detail);
    }

    return response.json();
  }

  async getDatasetIndexStatus(datasetId: string): Promise<DatasetIndexStatus> {
    const response = await this.fetchWithTimeout(
      `${API_BASE_URL}/search/dataset/index/status?dataset_id=${encodeURIComponent(datasetId)}`,
      {
        headers: this.getHeaders(false),
      },
      REQUEST_TIMEOUT_MS
    );

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao consultar status da indexacao');
      throw new Error(detail);
    }

    return response.json();
  }

  async indexDataset(
    datasetId: string,
    forceReindex = false,
    onStatus?: DatasetIndexStatusListener
  ): Promise<void> {
    let status = await this.getDatasetIndexStatus(datasetId);
    if (forceReindex || status.status === 'idle' || status.status === 'failed') {
      status = await this.startDatasetIndex(datasetId, forceReindex);
    }

    if (status.status === 'running') {
      onStatus?.(status);
    }

    while (status.status === 'running') {
      await new Promise((resolve) => window.setTimeout(resolve, INDEX_POLL_INTERVAL_MS));
      status = await this.getDatasetIndexStatus(datasetId);
      onStatus?.(status);
    }

    if (status.status === 'failed') {
      throw new Error(status.error || 'Falha na indexacao do dataset');
    }

    if (status.status !== 'completed') {
      throw new Error(`Status inesperado da indexacao: ${status.status}`);
    }
  }

  async searchDataset(query: string, datasetId: string, queryId?: string): Promise<SearchResponse> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/search/dataset`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        dataset_id: datasetId,
        query,
        query_id: queryId,
        mode: 'compare',
        top_k: 25,
      }),
    });

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro na busca no dataset');
      throw new Error(detail);
    }

    return response.json();
  }
  async listDatasets(): Promise<DatasetSummary[]> {
    const response = await this.fetchWithTimeout(`${API_BASE_URL}/datasets`, {
      headers: this.getHeaders(false),
    });

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao listar datasets');
      throw new Error(detail);
    }

    return response.json();
  }

  async getDataset(datasetId: string): Promise<DatasetDetail> {
    const response = await this.fetchWithTimeout(`${API_BASE_URL}/datasets/${datasetId}`, {
      headers: this.getHeaders(false),
    });

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao carregar dataset');
      throw new Error(detail);
    }

    return response.json();
  }

  async getEvaluationQueries(datasetId: string): Promise<EvaluationQuery[]> {
    const response = await this.fetchWithTimeout(
      `${API_BASE_URL}/evaluation/queries?dataset_id=${encodeURIComponent(datasetId)}`,
      { headers: this.getHeaders(false) }
    );

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao carregar queries de avaliação');
      throw new Error(detail);
    }

    return response.json();
  }

  async listBenchmarkLabels(datasetId: string): Promise<BenchmarkLabel[]> {
    const response = await this.fetchWithTimeout(`${API_BASE_URL}/benchmarks/labels?dataset_id=${encodeURIComponent(datasetId)}`, {
      headers: this.getHeaders(false),
    });

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao listar gabaritos');
      throw new Error(detail);
    }

    const payload = await response.json();
    return payload.items ?? [];
  }

  async upsertBenchmarkLabel(payload: BenchmarkLabelInput): Promise<BenchmarkLabel> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/benchmarks/labels`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao salvar gabarito');
      throw new Error(detail);
    }

    return response.json();
  }

  async deleteBenchmarkLabel(datasetId: string, benchmarkId: string): Promise<void> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/benchmarks/labels/${encodeURIComponent(datasetId)}/${encodeURIComponent(benchmarkId)}`, {
      method: 'DELETE',
      headers: this.getHeaders(false),
    });

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao remover gabarito');
      throw new Error(detail);
    }
  }

  async startBatchEvaluation(datasetId: string = 'beir/trec-covid'): Promise<BatchEvaluationStatus> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/benchmarks/evaluate/start`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        dataset_id: datasetId,
        pipelines: ['classical', 'quantum', 'statistical'],
      }),
    });

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao iniciar avaliação batch');
      throw new Error(detail);
    }

    return response.json();
  }

  async getBatchEvaluationStatus(): Promise<BatchEvaluationStatus> {
    const response = await this.fetchWithAuth(`${API_BASE_URL}/benchmarks/evaluate/status`, {
      headers: this.getHeaders(false),
    });

    if (!response.ok) {
      const detail = await this.readErrorDetail(response, 'Erro ao consultar status da avaliação');
      throw new Error(detail);
    }

    return response.json();
  }
}

export const api = new ApiClient();



