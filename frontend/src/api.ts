// Host-agnostic IPC primitives. These route to either the Tauri shell or the
// native .NET (Photino) shell automatically — see frontend/src/host. The rest
// of this file is unchanged and unaware of which backend answers.
import { invoke, listen, type UnlistenFn } from './host'

// ─── Network ────────────────────────────────────────────────────────────────

export type NetworkKind = 'feedforward' | 'next_token' | 'transformer'

export interface TransformerHParams {
  n_ctx:     number
  n_embd:    number
  n_layers:  number
  n_heads:   number
  n_ff:      number
  rope_theta: number
  rms_eps:   number
}
export type Activation = 'identity' | 'relu' | 'sigmoid' | 'tanh' | 'softmax'

export type Layer =
  | { type: 'linear'; in_dim: number; out_dim: number }
  | { type: 'activation'; activation: Activation }

export interface Network {
  id: string
  name: string
  kind: NetworkKind
  seed: number
  created_at: string
  trained: boolean
  /** Set to true once a pre-training run completes; gates fine-tuning. */
  pretrained?: boolean
  input_dim: number
  output_dim: number
  layers: Layer[]
  parameter_count: number
  hidden_layers?: Layer[] | null
  context_size?: number | null
  transformer?: TransformerHParams | null
}

export interface CreateNetworkRequest {
  name: string
  kind: NetworkKind
  seed: number
  layers: Layer[]
  input_dim?: number | null
  context_size?: number | null
  transformer?: TransformerHParams | null
}

export const networks = {
  create: (req: CreateNetworkRequest) => invoke<Network>('create_network', { req }),
  list:   ()                            => invoke<Network[]>('list_networks'),
  get:    (id: string)                  => invoke<Network>('get_network', { id }),
  delete: (id: string)                  => invoke<boolean>('delete_network', { id }),
}

// ─── Corpus ─────────────────────────────────────────────────────────────────

export interface FeedforwardCorpus {
  features: number[]
  targets: number[]
  rows: number
  in_dim: number
  out_dim: number
}

export interface FineTunePair { input: string; output: string }
export type Stage = 'pretrain' | 'finetune'

export interface SetCorpusRequest {
  network_id: string
  feedforward?: FeedforwardCorpus
  text?: string
  pairs?: FineTunePair[]
  stage?: Stage
  vocab_mode?: 'char' | 'word'
}

export interface Corpus {
  network_id: string
  kind: NetworkKind
  updated_at: string
  feedforward?: FeedforwardCorpus
  text?: string
  pairs?: FineTunePair[]
  stage?: Stage
}

export interface CorpusStats {
  kind: NetworkKind
  stage?: Stage
  rows?: number
  in_dim?: number
  out_dim?: number
  text_chars?: number
  text_tokens?: number
  pair_count?: number
  vocab_size?: number
  vocab_mode?: string
  training_examples?: number
  vocab_ready: boolean
  model_ready: boolean
}

export const corpus = {
  set:   (req: SetCorpusRequest)        => invoke<CorpusStats>('set_corpus', { req }),
  get:   (network_id: string)           => invoke<Corpus | null>('get_corpus', { networkId: network_id }),
  stats: (network_id: string)           => invoke<CorpusStats | null>('corpus_stats', { networkId: network_id }),
}

// ─── Vocabulary ─────────────────────────────────────────────────────────────

export type VocabMode = 'char' | 'subword' | 'word' | 'advanced'

export interface VocabularyOptions {
  subword_merges: number
  word_top_n: number
}

export interface VocabularyInfo {
  mode: VocabMode
  tokens: string[]
  options: VocabularyOptions
  updated_at: string
}

export const vocabulary = {
  build: (network_id: string, mode: Exclude<VocabMode, 'advanced'>, options: VocabularyOptions) =>
    invoke<VocabularyInfo>('build_vocabulary', {
      req: { network_id, mode, options },
    }),
  setAdvanced: (network_id: string, tokens: string[]) =>
    invoke<VocabularyInfo>('set_advanced_vocabulary', {
      req: { network_id, tokens },
    }),
  get: (network_id: string) =>
    invoke<VocabularyInfo | null>('get_vocabulary', { networkId: network_id }),
  tokenize: (network_id: string, text: string) =>
    invoke<[number, string][]>('tokenize_preview', { networkId: network_id, text }),
}

// ─── Training ───────────────────────────────────────────────────────────────

export type OptimizerKind = 'adam' | 'adamw' | 'lamb' | 'sgd'

export interface OptimizerConfig {
  kind: OptimizerKind
  lr: number
  beta1?: number
  beta2?: number
  eps?: number
  momentum?: number
  weight_decay?: number
}

export interface TrainingConfig {
  epochs: number
  batch_size: number
  optimizer: OptimizerConfig
  loss: 'mse' | 'crossentropy'
  seed: number
  mask_user_tokens?: boolean
  /** Names of transformer components to freeze during training.
   * Recognized: "embedding", "output", "output_norm", "block:N". */
  frozen_layers?: string[]
}

export interface TrainingRequest { network_id: string; config: TrainingConfig }

export interface TrainingStatus {
  training_id: string
  status: 'running' | 'completed' | 'cancelled' | 'error'
  epoch: number
  total_epochs: number
  last_loss: number
  loss_history: number[]
  elapsed_secs: number
}

export interface TrainingUpdate {
  training_id: string
  epoch: number
  total_epochs: number
  loss: number
  loss_history: number[]
  elapsed_secs: number
}

export interface TrainingFinished {
  training_id: string
  /** 'aborted' = cancelled with rollback (Abort button) */
  status: 'completed' | 'cancelled' | 'aborted'
  final_loss: number
  total_epochs: number
  elapsed_secs: number
}

export interface TrainingError { training_id: string; message: string }

// ─── Training history ────────────────────────────────────────────────────────

export interface TrainingConfigSummary {
  optimizer: string
  lr: number
  batch_size: number
  epochs: number
}

export interface TrainingRun {
  id: string
  network_id: string
  started_at: string
  finished_at: string
  status: 'completed' | 'cancelled' | 'aborted' | 'error'
  config_summary: TrainingConfigSummary
  total_epochs: number
  epochs_run: number
  final_loss: number
  elapsed_secs: number
  loss_history: number[]
}

export const training = {
  start:  (req: TrainingRequest) =>
    invoke<{ training_id: string; status: string }>('start_training', { req }),
  /** Graceful stop: keeps whatever weights the model has at this moment. */
  stop:   (trainingId: string) =>
    invoke<boolean>('stop_training', { trainingId }),
  /** Cancel and roll back to the pre-training weights. */
  abort:  (trainingId: string) =>
    invoke<boolean>('abort_training', { trainingId }),
  status: (trainingId: string) =>
    invoke<TrainingStatus>('get_training_status', { trainingId }),

  history:      (networkId: string) =>
    invoke<TrainingRun[]>('get_training_history', { networkId }),
  clearHistory: (networkId: string) =>
    invoke<void>('clear_training_history', { networkId }),

  onUpdate: (handler: (u: TrainingUpdate) => void): Promise<UnlistenFn> =>
    listen<TrainingUpdate>('training_update', (e) => handler(e.payload)),
  onFinished: (handler: (r: TrainingFinished) => void): Promise<UnlistenFn> =>
    listen<TrainingFinished>('training_finished', (e) => handler(e.payload)),
  onError: (handler: (err: TrainingError) => void): Promise<UnlistenFn> =>
    listen<TrainingError>('training_error', (e) => handler(e.payload)),
}

// ─── Inference ──────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
}

export interface InferRequest {
  network_id: string
  features?: number[]
  prompt?: string
  max_new_tokens?: number
  temperature?: number
  /** Full chat history for multi-turn conversations. The last message must
   *  be `role: 'user'`; the backend appends `<assistant>` and generates. */
  messages?: ChatMessage[]
}

export interface InferResponse {
  /** Feed-forward synchronous result. */
  output?: number[]
  /** Next-token streaming job id; subscribe to inference events. */
  inference_id?: string
}

export interface InferenceToken {
  inference_id: string
  index: number
  token: string
  probability: number
}

export interface InferenceFinished {
  inference_id: string
  status: 'completed' | 'cancelled'
  generated: string
  token_count: number
}

export interface InferenceError { inference_id: string; message: string }

// ─── Export ─────────────────────────────────────────────────────────────────

export type ExportFormat = 'pytorch' | 'onnx' | 'gguf'

export interface ExportPayload {
  format: ExportFormat
  filename: string
  data_b64: string
  size_bytes: number
}

export const exporter = {
  /** Server-side export to bytes. The frontend converts the base64 payload
   *  into a Blob and triggers a browser download. */
  run: (network_id: string, format: ExportFormat) =>
    invoke<ExportPayload>('export_network', { networkId: network_id, format }),
}

// ─── Server ─────────────────────────────────────────────────────────────────

export interface ServerPermissions {
  allow_list: boolean
  allow_inference: boolean
  allow_export: boolean
  allow_upload: boolean
  allow_train: boolean
  allow_create: boolean
  allow_delete: boolean
}

export interface ServerConfig {
  id: string
  name: string
  port: number
  localhost_only: boolean
  auth_token: string
  permissions: ServerPermissions
  auto_start: boolean
  created_at: string
}

export interface ServerSummary extends ServerConfig {
  running: boolean
  request_count: number
  last_error: string | null
}

export interface CreateServerRequest {
  name: string
  port: number
  localhost_only?: boolean
  auth_token?: string
  permissions?: ServerPermissions
}

export interface UpdateServerRequest {
  id: string
  name?: string
  port?: number
  localhost_only?: boolean
  auth_token?: string
  permissions?: ServerPermissions
  auto_start?: boolean
}

export const servers = {
  list:   ()                          => invoke<ServerSummary[]>('list_servers'),
  create: (req: CreateServerRequest)  => invoke<ServerSummary>('create_server', { req }),
  update: (req: UpdateServerRequest)  => invoke<ServerSummary>('update_server', { req }),
  delete: (id: string)                => invoke<boolean>('delete_server', { id }),
  start:  (id: string)                => invoke<ServerSummary>('start_server', { id }),
  stop:   (id: string)                => invoke<ServerSummary>('stop_server', { id }),
  status: (id: string)                => invoke<ServerSummary>('server_status', { id }),
}

export interface InferActivations {
  layer_names: string[]
  activations: number[][]
  dims: number[]
}

export const inference = {
  run:    (req: InferRequest) => invoke<InferResponse>('infer', { req }),
  /** Forward pass that also returns activations after every layer. Feed-forward only. */
  runWithActivations: (network_id: string, features: number[]) =>
    invoke<InferActivations>('infer_with_activations', { networkId: network_id, features }),
  stop:   (inferenceId: string) => invoke<boolean>('stop_inference', { inferenceId }),

  onToken:    (handler: (t: InferenceToken) => void): Promise<UnlistenFn> =>
    listen<InferenceToken>('inference_token', (e) => handler(e.payload)),
  onFinished: (handler: (r: InferenceFinished) => void): Promise<UnlistenFn> =>
    listen<InferenceFinished>('inference_finished', (e) => handler(e.payload)),
  onError:    (handler: (e: InferenceError) => void): Promise<UnlistenFn> =>
    listen<InferenceError>('inference_error', (e) => handler(e.payload)),
}
