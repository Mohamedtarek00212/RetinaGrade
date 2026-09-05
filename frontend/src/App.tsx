import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  CircleHelp,
  Clock3,
  GitCompareArrows,
  ImagePlus,
  LockKeyhole,
  LoaderCircle,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  ShieldAlert,
  Upload,
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

import { predictOnDevice } from "./inference/model";
import type { BrowserPrediction, ModelProgress } from "./inference/types";

const ACCEPTED_TYPES = ["image/png", "image/jpeg"];
const MAX_FILE_SIZE = 15 * 1024 * 1024;

const GRADE_DETAILS = [
  { short: "No DR", description: "No visible diabetic retinopathy" },
  { short: "Mild", description: "Mild diabetic retinopathy" },
  { short: "Moderate", description: "Moderate diabetic retinopathy" },
  { short: "Severe", description: "Severe diabetic retinopathy" },
  { short: "PDR", description: "Proliferative diabetic retinopathy" },
];

const SAMPLES = [
  { name: "Study sample A", path: "/samples/no-dr.png" },
  { name: "Study sample B", path: "/samples/moderate.png" },
  { name: "Study sample C", path: "/samples/pdr.png" },
];

function confidenceBand(confidence: number) {
  if (confidence >= 0.75) return { label: "High", note: "The model strongly favors this grade." };
  if (confidence >= 0.5) return { label: "Moderate", note: "Review the nearby grade probabilities." };
  return { label: "Low", note: "The prediction is uncertain across multiple grades." };
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<BrowserPrediction | null>(null);
  const [modelProgress, setModelProgress] = useState<ModelProgress>({
    label: "On-device model",
  });
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [resultView, setResultView] = useState<"summary" | "thresholds">("summary");

  const derived = prediction
    ? (() => {
        const ranked = prediction.probabilities
          .map((probability, grade) => ({ probability, grade }))
          .sort((left, right) => right.probability - left.probability);
        return {
          referableProbability: prediction.probabilities.slice(2).reduce((sum, value) => sum + value, 0),
          expectedGrade: prediction.probabilities.reduce(
            (sum, probability, grade) => sum + probability * grade,
            0,
          ),
          alternative: ranked.find((item) => item.grade !== prediction.grade) ?? ranked[0],
          confidence: confidenceBand(prediction.confidence),
        };
      })()
    : null;

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const selectFile = (selected: File) => {
    setError(null);
    setPrediction(null);
    setResultView("summary");
    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setError("Choose a PNG or JPEG fundus image.");
      return;
    }
    if (selected.size > MAX_FILE_SIZE) {
      setError("The image is larger than the 15 MB limit.");
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleInput = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    if (selected) selectFile(selected);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const selected = event.dataTransfer.files[0];
    if (selected) selectFile(selected);
  };

  const loadSample = async (name: string, path: string) => {
    const response = await fetch(path);
    const blob = await response.blob();
    selectFile(new File([blob], `${name}.png`, { type: "image/png" }));
  };

  const clearImage = () => {
    setFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const analyze = async () => {
    if (!file) return;
    setIsPredicting(true);
    setError(null);
    try {
      const result = await predictOnDevice(file, setModelProgress);
      setPrediction(result);
      setModelProgress({ label: `Model ready · ${result.runtime}` });
      setResultView("summary");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Analysis failed.");
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><ScanLine size={22} /></span>
          <div>
            <strong>RetinaGrade</strong>
            <span>Dual-SwinOrd Research Demo</span>
          </div>
        </div>
        <div className="service-status ready">
          <span className="status-dot" />
          {modelProgress.label}
          {modelProgress.percent !== undefined ? ` · ${modelProgress.percent}%` : ""}
        </div>
      </header>

      <main>
        <section className="workspace-heading">
          <div>
            <p className="eyebrow">Fundus image assessment</p>
            <h1>Diabetic retinopathy grading</h1>
          </div>
          <div className="model-metrics" aria-label="Model test metrics">
            <span><strong>86.84%</strong> Test accuracy</span>
            <span><strong>0.9074</strong> Test QWK</span>
          </div>
        </section>

        <section className="workspace-grid">
          <div className="workspace-panel image-panel">
            <div className="panel-heading">
              <div>
                <span className="step">01</span>
                <h2>Fundus image</h2>
              </div>
              {file && (
                <button className="icon-button" type="button" onClick={clearImage} title="Clear image">
                  <RotateCcw size={18} />
                </button>
              )}
            </div>

            {preview ? (
              <div className="image-preview">
                <img src={preview} alt="Selected retinal fundus" />
                <div className="image-meta">
                  <span>{file?.name}</span>
                  <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : ""}</span>
                </div>
              </div>
            ) : (
              <div
                className={`dropzone ${isDragging ? "dragging" : ""}`}
                onDragEnter={() => setIsDragging(true)}
                onDragLeave={() => setIsDragging(false)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
              >
                <span className="upload-icon"><ImagePlus size={28} /></span>
                <strong>Drop a fundus image here</strong>
                <span>or choose a PNG or JPEG file</span>
                <button type="button" className="secondary-button">
                  <Upload size={17} /> Choose image
                </button>
              </div>
            )}
            <input
              ref={inputRef}
              className="visually-hidden"
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleInput}
            />

            {!preview && (
              <div className="sample-section">
                <span>Try a study sample</span>
                <div className="sample-list">
                  {SAMPLES.map((sample) => (
                    <button key={sample.path} type="button" onClick={() => loadSample(sample.name, sample.path)}>
                      <img src={sample.path} alt="" />
                      {sample.name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div className="error-message" role="alert">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}

            <button
              className="primary-button"
              type="button"
              disabled={!file || isPredicting}
              onClick={analyze}
            >
              {isPredicting ? <LoaderCircle className="spin" size={19} /> : <ScanLine size={19} />}
              {isPredicting ? "Analyzing image..." : "Analyze image"}
            </button>
            <div className="privacy-note">
              <LockKeyhole size={14} />
              <span>Runs on this device. The image never leaves your browser.</span>
            </div>
          </div>

          <div className="workspace-panel result-panel">
            <div className="panel-heading">
              <div>
                <span className="step">02</span>
                <h2>Model assessment</h2>
              </div>
              <CircleHelp size={18} className="muted-icon" aria-label="Research prediction" />
            </div>

            {prediction ? (
              <div className="prediction-content">
                <div className={`grade-summary grade-${prediction.grade}`}>
                  <div className="grade-number">
                    <span>Grade</span>
                    <strong>{prediction.grade}</strong>
                  </div>
                  <div className="grade-label">
                    <span>Predicted severity</span>
                    <h3>{prediction.label}</h3>
                    <p>{GRADE_DETAILS[prediction.grade]?.description}</p>
                  </div>
                  <div className="confidence">
                    <span>Confidence</span>
                    <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
                  </div>
                </div>

                {prediction.assessment_status && prediction.confidence_margin !== undefined && (
                  <div className={`assessment-banner ${prediction.assessment_status}`}>
                    {prediction.assessment_status === "conclusive" ? (
                      <CheckCircle2 size={18} />
                    ) : (
                      <AlertCircle size={18} />
                    )}
                    <div>
                      <strong>
                        {prediction.assessment_status === "conclusive"
                          ? "Clear model preference"
                          : "Review recommended"}
                      </strong>
                      <span>
                        Top-two confidence margin: {(prediction.confidence_margin * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                )}

                <div className="result-tabs" role="tablist" aria-label="Result views">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={resultView === "summary"}
                    onClick={() => setResultView("summary")}
                  >
                    Summary
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={resultView === "thresholds"}
                    onClick={() => setResultView("thresholds")}
                  >
                    Threshold details
                  </button>
                </div>

                {resultView === "summary" ? (
                  <>
                    <div className="probability-section">
                      <div className="section-label">
                        <span>Class probabilities</span>
                        <Activity size={16} />
                      </div>
                      <div className="probability-list">
                        {prediction.probabilities.map((probability, grade) => (
                          <div className="probability-row" key={GRADE_DETAILS[grade].short}>
                            <div className="probability-label">
                              <span className={`grade-dot grade-${grade}`} />
                              <span>{grade} · {GRADE_DETAILS[grade].short}</span>
                              <strong>{(probability * 100).toFixed(1)}%</strong>
                            </div>
                            <div className="probability-track">
                              <span className={`probability-fill grade-${grade}`} style={{ width: `${probability * 100}%` }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {derived && (
                      <div className="insight-section">
                        <div className="section-label">
                          <span>Derived insights</span>
                          <BarChart3 size={16} />
                        </div>
                        <div className="insight-grid">
                          <div className="insight-item">
                            <span>Referable DR likelihood</span>
                            <strong>{(derived.referableProbability * 100).toFixed(1)}%</strong>
                            <small>Combined probability of Grades 2-4</small>
                          </div>
                          <div className="insight-item">
                            <span>Expected grade</span>
                            <strong>{derived.expectedGrade.toFixed(2)}</strong>
                            <small>Probability-weighted severity</small>
                          </div>
                          <div className="insight-item">
                            <span>Confidence level</span>
                            <strong>{derived.confidence.label}</strong>
                            <small>{derived.confidence.note}</small>
                          </div>
                          <div className="insight-item">
                            <span>Closest alternative</span>
                            <strong>Grade {derived.alternative.grade}</strong>
                            <small>
                              {GRADE_DETAILS[derived.alternative.grade].short} · {(derived.alternative.probability * 100).toFixed(1)}%
                            </small>
                          </div>
                        </div>
                      </div>
                    )}

                    {prediction.quality && prediction.inference_ms !== undefined && (
                      <div className="quality-strip">
                        <div>
                          <ShieldCheck size={17} />
                          <span>
                            <strong>Image quality passed</strong>
                            {prediction.quality.width} × {prediction.quality.height}px
                          </span>
                        </div>
                        <div>
                          <Clock3 size={17} />
                          <span>
                            <strong>Model inference</strong>
                            {prediction.inference_ms.toFixed(0)} ms
                          </span>
                        </div>
                      </div>
                    )}
                    {prediction.quality && prediction.quality.warnings.length > 0 && (
                      <div className="quality-warning">
                        <AlertCircle size={16} />
                        <span>{prediction.quality.warnings.join(" ")}</span>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="ordinal-section">
                      <div className="section-label">
                        <span>Ordinal threshold analysis</span>
                        <GitCompareArrows size={16} />
                      </div>
                      <div className="ordinal-list">
                        {prediction.ordinal_probabilities.map((probability, threshold) => (
                          <div className="ordinal-row" key={threshold}>
                            <div>
                              <span>Probability grade is above {threshold}</span>
                              <strong>{(probability * 100).toFixed(1)}%</strong>
                            </div>
                            <div className="ordinal-track">
                              <span style={{ width: `${probability * 100}%` }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="interpretation-note">
                      <h4>How to read this output</h4>
                      <p>
                        The predicted grade is the class with the highest probability. A Grade 2-4
                        probability is summarized as referable DR likelihood for research comparison;
                        it is not a clinical referral decision. Confidence describes model certainty,
                        not guaranteed correctness.
                      </p>
                    </div>
                  </>
                )}

                <div className="result-note">
                  <CheckCircle2 size={18} />
                  <p>
                    Analysis completed locally with the validation-selected ONNX model on {prediction.runtime}.
                  </p>
                </div>
              </div>
            ) : (
              <div className="empty-result">
                <span className="scan-placeholder"><ScanLine size={38} /></span>
                <h3>Awaiting an image</h3>
                <p>Your grade, confidence, and full probability distribution will appear here.</p>
                <div className="grade-scale" aria-label="Diabetic retinopathy grade scale">
                  {GRADE_DETAILS.map((grade, index) => (
                    <div key={grade.short}>
                      <span className={`grade-dot grade-${index}`} />
                      <strong>{index}</strong>
                      <small>{grade.short}</small>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        <aside className="medical-disclaimer">
          <ShieldAlert size={19} />
          <p><strong>Research use only.</strong> RetinaGrade is not a medical device and must not be used for diagnosis, triage, or treatment decisions.</p>
        </aside>
      </main>
    </div>
  );
}

export default App;
