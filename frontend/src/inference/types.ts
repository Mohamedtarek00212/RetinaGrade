export type ImageQuality = {
  width: number;
  height: number;
  brightness: number;
  sharpness: number;
  visible_area_ratio: number;
  retinal_color_score: number;
  acceptable: boolean;
  warnings: string[];
};

export type BrowserPrediction = {
  grade: number;
  label: string;
  confidence: number;
  probabilities: number[];
  ordinal_probabilities: number[];
  assessment_status: "conclusive" | "review_recommended";
  confidence_margin: number;
  inference_ms: number;
  quality: ImageQuality;
  runtime: "WebGPU" | "CPU";
};

export type ModelProgress = {
  label: string;
  percent?: number;
};
