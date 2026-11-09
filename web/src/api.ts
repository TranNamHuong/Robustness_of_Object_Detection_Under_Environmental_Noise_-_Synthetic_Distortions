export type Experiment = {
  id: string;
  title: string;
  dataset: string;
  dataset_kind: string;
  protocol: string;
  source: string;
  kind: string;
  available: boolean;
  record_count?: number;
  model_count?: number;
  map50_count?: number;
  map50_95_count?: number;
  models?: string[];
  conditions?: string[];
  stages?: string[];
  reference_only?: boolean;
};

export type MetricRecord = {
  experiment_id: string;
  protocol_id: string;
  dataset: string;
  dataset_kind: string;
  protocol: string;
  model: string;
  model_version: string;
  condition: string;
  condition_label_bilingual: string;
  family: string;
  severity: number | null;
  stage: string;
  mAP50: number | null;
  mAP50_95: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  tp: number | null;
  fp: number | null;
  fn: number | null;
  latency_ms: number | null;
  fps: number | null;
  images: number | null;
  ground_truth_boxes: number | null;
  classes_scored: number | null;
  input_size: number | null;
  confidence_threshold: number | null;
  matching_iou: number | null;
  metric_implementation: string | null;
  derived_fields: string[];
  source_file: string;
  reference_only?: boolean;
  subset_count?: number | null;
  mAP50_std?: number | null;
  mAP50_95_std?: number | null;
  precision_std?: number | null;
  recall_std?: number | null;
  relative_change_vs_clean_pct?: number | null;
  relative_change_vs_clean_pct_std?: number | null;
};

export type MetricsResponse = {
  experiment: Experiment | null;
  filters: { models: string[]; conditions: string[]; families: string[]; stages: string[] };
  records: MetricRecord[];
  record_count: number;
  source_file: string | null;
  reference_only: boolean;
};

export type GalleryItem = {
  image_id: string; image_url: string; image_size: { width: number; height: number }; class_names: string[];
  ground_truth: { class_name: string; x1: number; y1: number; x2: number; y2: number; group_of: boolean }[];
  predictions: { class_name: string; confidence: number; x1: number; y1: number; x2: number; y2: number }[];
  source: string; license: string; author: string; dataset: string; model: string; prediction_threshold: number;
};
export type GalleryResponse = { items: GalleryItem[]; page: number; page_size: number; total: number; classes: string[]; protocol: string };
export type RecommendationCandidate = { model: string; value: number; metric: string; condition?: string; source_file?: string; protocol?: string; images?: number; clean_mAP50_95?: number; noisy_mAP50_95?: number; raw_mAP50_95?: number; restored_mAP50_95?: number };
export type RecommendationResponse = { status: string; criterion?: string; winner?: RecommendationCandidate; candidates: RecommendationCandidate[]; experiment?: Experiment; reason?: string; rule?: string };

export type HealthResponse = {
  status: string;
  device: string;
  cuda_available: boolean;
  gpu_name: string | null;
  torch_version: string;
  torchvision_version: string | null;
  ultralytics_version: string;
  models: Record<string, boolean | string>;
};

export type Detection = { class_id: number; label: string; confidence: number; bbox_xyxy: number[] };
export type DetectionView = { detections: Detection[]; latency_ms: number; device: string };
export type DetectionResponse = {
  source_filename: string;
  image_size: { width: number; height: number };
  condition: string;
  family: string;
  severity: number;
  seed: number;
  recovery_method: string | null;
  confidence_threshold: number;
  nms_iou_threshold: number;
  has_ground_truth: boolean;
  metrics: null | Record<string, number>;
  models: Record<string, { model: string; views: Record<string, DetectionView>; images: { original: string; noisy: string; restored: string | null } }>;
  provenance: Record<string, string>;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
    } catch { /* keep the HTTP fallback */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),
  experiments: () => request<Experiment[]>('/api/experiments'),
  metrics: (experimentId: string) => request<MetricsResponse>(`/api/metrics?experiment_id=${encodeURIComponent(experimentId)}`),
  gallery: (page: number, className: string) => request<GalleryResponse>(`/api/gallery?page=${page}&page_size=12${className ? `&class_name=${encodeURIComponent(className)}` : ''}`),
  recommendations: (experimentId: string, criterion: string, condition: string, stage?: string) => request<RecommendationResponse>(`/api/recommendations?experiment_id=${encodeURIComponent(experimentId)}&criterion=${criterion}${condition ? `&condition=${encodeURIComponent(condition)}` : ''}${stage ? `&stage=${encodeURIComponent(stage)}` : ''}`),
  detect: (form: FormData) => request<DetectionResponse>('/api/detect', { method: 'POST', body: form }),
  compare: (form: FormData) => request<DetectionResponse>('/api/compare', { method: 'POST', body: form }),
  exportUrl: (experimentId: string) => `/api/exports?experiment_id=${encodeURIComponent(experimentId)}`,
};

export function formatMetric(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: digits });
}
