import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PredictionResult {
  post_id: string | number;
  viral: number;
  probability_viral: number;
  probability_not_viral: number;
}

export interface TrainingMetrics {
  test_f1_score: number;
  training_samples: number;
  test_samples: number;
  top_features: string[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelStatus {
  trained: boolean;
  features_count: number;
  top_features: string[];
}

export interface PredictRequest {
  Post_ID?: string;
  Pseudo_Caption?: string;
  Post_Date?: string;
  Platform: string;
  Hashtag?: string;
  Content_Type: string;
  Region: string;
  Views: number;
  Likes: number;
  Shares: number;
  Comments: number;
}

export interface BestDayResult {
  platform: string;
  content_type: string;
  best_day: string;
  best_day_number: number;
  viral_rate: number;
  post_count: number;
}

export interface HeatmapCell {
  platform: string;
  content_type: string;
  day: string;
  day_number: number;
  viral_rate: number;
  post_count: number;
}

export interface BestDayAnalysis {
  best_days_by_combination: BestDayResult[];
  overall_best_day: string;
  overall_best_day_number: number;
  platforms: string[];
  content_types: string[];
  heatmap_data: HeatmapCell[];
  total_posts_analyzed: number;
}

export interface ShapContribution {
  feature: string;
  feature_raw: string;
  value: number | string;
  shap_value: number;
  impact: 'positive' | 'negative' | 'neutral';
}

export interface ShapExplanation {
  base_value: number;
  contributions: ShapContribution[];
  top_positive: ShapContribution[];
  top_negative: ShapContribution[];
}

export interface ExplainedPrediction {
  viral: number;
  probability_viral: number;
  probability_not_viral: number;
  explanation: ShapExplanation | null;
  feature_values: Record<string, number | string>;
}

export const getModelStatus = async (): Promise<ModelStatus> => {
  const response = await api.get('/model/status');
  return response.data;
};

export const trainModel = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/train', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const predict = async (data: PredictRequest | PredictRequest[]) => {
  const payload = Array.isArray(data) ? { data } : data;
  const response = await api.post('/predict', payload);
  return response.data;
};

export const predictFromFile = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const analyzeBestDay = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/analyze/best-day', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getBestDays = async () => {
  const response = await api.get('/model/best-days');
  return response.data;
};

export const explainPrediction = async (data: PredictRequest) => {
  const response = await api.post('/explain', data);
  return response.data;
};

// Insights data interfaces
export interface InsightsData {
  totalPosts: number;
  viralPosts: number;
  viralRate: number;
  bestHashtag: string;
  bestHashtagRate: number;
  bestDay: string;
  bestDayRate: number;
  platformData: Array<{ name: string; posts: number; viralRate: number }>;
  contentTypeData: Array<{ name: string; posts: number; viralRate: number; color: string }>;
  hashtagData: Array<{ hashtag: string; posts: number; viralRate: number }>;
  dayOfWeekData: Array<{ day: string; dayNum: number; avgViralRate: number }>;
  regionData: Array<{ name: string; value: number; viral: number }>;
  correlationData: Array<{ feature: string; correlation: number }>;
  bestDayHeatmap: Array<{ platform: string; contentType: string; bestDay: string; viralRate: number }>;
}

export const getInsightsData = async (): Promise<InsightsData> => {
  const response = await api.get('/insights/data');
  return response.data;
};
