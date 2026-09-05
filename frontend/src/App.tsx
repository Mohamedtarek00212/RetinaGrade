import {
  Activity,
  AlertCircle,
  BarChart3,
  Camera,
  CheckCircle2,
  CircleHelp,
  ClipboardCheck,
  Clock3,
  FileDown,
  FileText,
  Eye,
  GitCompareArrows,
  ImagePlus,
  LockKeyhole,
  Lightbulb,
  LoaderCircle,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  ShieldAlert,
  Stethoscope,
  Upload,
  UserRound,
} from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

import { predictOnDevice, prepareOnDeviceModel } from "./inference/model";
import type { BrowserPrediction, ModelProgress } from "./inference/types";
import { generateDoctorReport, generatePatientReport } from "./report";
import type { ReportDetails } from "./report";

const ACCEPTED_TYPES = ["image/png", "image/jpeg"];
const MAX_FILE_SIZE = 15 * 1024 * 1024;
type ResultView = "summary" | "explanation" | "thresholds";

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

function guidanceFor(prediction: BrowserPrediction) {
  const guidance: { title: string; text: string; kind: "quality" | "review" | "screening" }[] = [];
  if (prediction.quality.warnings.length > 0) {
    guidance.push({
      title: "Consider another capture",
      text: "The quality checks found a possible limitation. A sharper, evenly illuminated fundus image may produce a more dependable model output.",
      kind: "quality",
    });
  } else {
    guidance.push({
      title: "Image quality accepted",
      text: "The automated checks accepted the image for size, exposure, visible area, and retinal color profile.",
      kind: "quality",
    });
  }

  if (prediction.assessment_status === "review_recommended") {
    guidance.push({
      title: "Model result is uncertain",
      text: "Nearby grades have similar probabilities. Do not use this output to make a care decision; obtain professional review.",
      kind: "review",
    });
  } else {
    guidance.push({
      title: "Clear model preference",
      text: "The leading grade is separated from the alternatives, but confidence describes model consistency rather than guaranteed correctness.",
      kind: "review",
    });
  }

  guidance.push(
    prediction.grade === 0
      ? {
          title: "Continue routine eye care",
          text: "No DR is the model's leading class. This does not rule out disease or replace the eye examinations recommended by a qualified professional.",
          kind: "screening",
        }
      : {
          title: "Professional review is appropriate",
          text: "The model favored a non-zero DR grade. A qualified eye-care professional must evaluate the image; this research demo cannot determine diagnosis or urgency.",
          kind: "screening",
        },
  );
  return guidance;
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
  const [resultView, setResultView] = useState<ResultView>("summary");
  const [reportDetails, setReportDetails] = useState<ReportDetails>({
    patientName: "",
    patientId: "",
    clinicianName: "",
    clinicName: "",
    clinicianNotes: "",
    patientNextSteps: "",
  });
  const [reportReviewed, setReportReviewed] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [generatingReport, setGeneratingReport] = useState<"doctor" | "patient" | null>(null);

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
    setReportReviewed(false);
    setReportError(null);
    setReportDetails((current) => ({
      ...current,
      patientName: "",
      patientId: "",
      clinicianNotes: "",
      patientNextSteps: "",
    }));
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
    setModelProgress({ label: "Preparing on-device model" });
    void prepareOnDeviceModel(setModelProgress)
      .then((runtime) => setModelProgress({ label: `Model ready · ${runtime}` }))
      .catch(() => setModelProgress({ label: "On-device model" }));
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
    setReportReviewed(false);
    setReportError(null);
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

  const updateReportField = (field: keyof ReportDetails, value: string) => {
    setReportDetails((current) => ({ ...current, [field]: value }));
    setReportError(null);
  };

  const downloadReport = async (audience: "doctor" | "patient") => {
    if (!file || !prediction) return;
    if (!reportDetails.patientName.trim() || !reportDetails.clinicianName.trim()) {
      setReportError("Patient name and reviewing clinician are required for both reports.");
      return;
    }
    if (audience === "patient" && !reportDetails.patientNextSteps.trim()) {
      setReportError("Enter the clinician-approved next steps before creating the patient summary.");
      return;
    }
    if (audience === "patient" && !reportReviewed) {
      setReportError("The clinician must confirm review before creating the patient summary.");
      return;
    }

    setGeneratingReport(audience);
    setReportError(null);
    try {
      const context = { file, prediction, details: reportDetails };
      if (audience === "doctor") await generateDoctorReport(context);
      else await generatePatientReport(context);
    } catch (reportGenerationError) {
      setReportError(
        reportGenerationError instanceof Error
          ? reportGenerationError.message
          : "Could not create the PDF report.",
      );
    } finally {
      setGeneratingReport(null);
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
                    aria-selected={resultView === "explanation"}
                    onClick={() => setResultView("explanation")}
                  >
                    Explanation
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={resultView === "thresholds"}
                    onClick={() => setResultView("thresholds")}
                  >
                    Thresholds
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
                ) : resultView === "explanation" && derived ? (
                  <div className="explanation-section">
                    <div className="section-label">
                      <span>Class activation map</span>
                      <Eye size={16} />
                    </div>
                    <figure className="focus-figure">
                      <img
                        src={prediction.explanation.image_url}
                        alt={`Model contribution map for the Grade ${prediction.grade} prediction`}
                      />
                      <figcaption>
                        <div className="focus-legend" aria-label="Contribution intensity scale">
                          <span>Lower contribution</span>
                          <i />
                          <span>Higher contribution</span>
                        </div>
                        <p>
                          Warmer areas contributed more strongly to the selected classification.
                          This map does not identify lesions or confirm disease.
                        </p>
                      </figcaption>
                    </figure>

                    <div className="explanation-copy">
                      <div className="section-label">
                        <span>Why the model chose this grade</span>
                        <Lightbulb size={16} />
                      </div>
                      <p>
                        Grade {prediction.grade} ({GRADE_DETAILS[prediction.grade].short}) received{" "}
                        {(prediction.confidence * 100).toFixed(1)}% probability. The closest
                        alternative was Grade {derived.alternative.grade} ({GRADE_DETAILS[derived.alternative.grade].short}) at{" "}
                        {(derived.alternative.probability * 100).toFixed(1)}%, producing a{" "}
                        {(prediction.confidence_margin * 100).toFixed(1)}% top-two margin.
                      </p>
                    </div>

                    <div className="guidance-section">
                      <div className="section-label">
                        <span>Research guidance</span>
                        <Camera size={16} />
                      </div>
                      <div className="guidance-list">
                        {guidanceFor(prediction).map((item) => (
                          <div className={`guidance-item ${item.kind}`} key={item.title}>
                            {item.kind === "quality" ? <Camera size={17} /> : item.kind === "review" ? <Lightbulb size={17} /> : <ShieldAlert size={17} />}
                            <span>
                              <strong>{item.title}</strong>
                              <small>{item.text}</small>
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
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

                <section className="report-section" aria-labelledby="report-heading">
                  <div className="section-label">
                    <span id="report-heading">Professional reports</span>
                    <FileText size={16} />
                  </div>

                  <div className="report-form">
                    <label>
                      <span>Patient name *</span>
                      <input
                        type="text"
                        value={reportDetails.patientName}
                        maxLength={80}
                        autoComplete="name"
                        onChange={(event) => updateReportField("patientName", event.target.value)}
                      />
                    </label>
                    <label>
                      <span>Patient ID</span>
                      <input
                        type="text"
                        value={reportDetails.patientId}
                        maxLength={40}
                        onChange={(event) => updateReportField("patientId", event.target.value)}
                      />
                    </label>
                    <label>
                      <span>Reviewing clinician *</span>
                      <input
                        type="text"
                        value={reportDetails.clinicianName}
                        maxLength={80}
                        onChange={(event) => updateReportField("clinicianName", event.target.value)}
                      />
                    </label>
                    <label>
                      <span>Clinic or practice</span>
                      <input
                        type="text"
                        value={reportDetails.clinicName}
                        maxLength={80}
                        onChange={(event) => updateReportField("clinicName", event.target.value)}
                      />
                    </label>
                    <label className="report-field-wide">
                      <span>Clinician notes</span>
                      <textarea
                        value={reportDetails.clinicianNotes}
                        maxLength={900}
                        rows={4}
                        onChange={(event) => updateReportField("clinicianNotes", event.target.value)}
                      />
                    </label>
                    <label className="report-field-wide">
                      <span>Clinician-approved next steps for the patient *</span>
                      <textarea
                        value={reportDetails.patientNextSteps}
                        maxLength={600}
                        rows={3}
                        onChange={(event) => updateReportField("patientNextSteps", event.target.value)}
                      />
                    </label>
                  </div>

                  <label className="review-confirmation">
                    <input
                      type="checkbox"
                      checked={reportReviewed}
                      onChange={(event) => {
                        setReportReviewed(event.target.checked);
                        setReportError(null);
                      }}
                    />
                    <span>
                      <strong>Clinician review confirmed</strong>
                      I reviewed the image and model output. The patient summary reflects my own
                      communication plan, not an automated care decision.
                    </span>
                  </label>

                  {reportError && (
                    <div className="report-error" role="alert">
                      <AlertCircle size={17} />
                      <span>{reportError}</span>
                    </div>
                  )}

                  <div className="report-actions">
                    <article>
                      <span className="report-type-icon"><Stethoscope size={20} /></span>
                      <div>
                        <strong>Clinician review report</strong>
                        <small>Detailed model output, image quality, focus map, notes, and sign-off.</small>
                      </div>
                      <button
                        type="button"
                        onClick={() => downloadReport("doctor")}
                        disabled={generatingReport !== null}
                      >
                        {generatingReport === "doctor" ? <LoaderCircle className="spin" size={17} /> : <FileDown size={17} />}
                        Download clinician PDF
                      </button>
                    </article>
                    <article>
                      <span className="report-type-icon patient"><UserRound size={20} /></span>
                      <div>
                        <strong>Patient information summary</strong>
                        <small>Plain-language result and the next steps approved by the clinician.</small>
                      </div>
                      <button
                        type="button"
                        onClick={() => downloadReport("patient")}
                        disabled={generatingReport !== null}
                      >
                        {generatingReport === "patient" ? <LoaderCircle className="spin" size={17} /> : <ClipboardCheck size={17} />}
                        Download patient PDF
                      </button>
                    </article>
                  </div>

                  <div className="report-privacy">
                    <LockKeyhole size={14} />
                    Report data and PDF generation remain inside this browser.
                  </div>
                </section>
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
