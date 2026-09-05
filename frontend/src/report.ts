import type { jsPDF as JsPDF } from "jspdf";

import type { BrowserPrediction } from "./inference/types";

export type ReportDetails = {
  patientName: string;
  patientId: string;
  clinicianName: string;
  clinicName: string;
  clinicianNotes: string;
  patientNextSteps: string;
};

type ReportContext = {
  file: File;
  prediction: BrowserPrediction;
  details: ReportDetails;
};

const COLORS = {
  ink: [35, 38, 42] as const,
  muted: [104, 111, 118] as const,
  line: [218, 222, 225] as const,
  paper: [248, 249, 249] as const,
  orange: [232, 106, 51] as const,
  teal: [37, 145, 134] as const,
  red: [169, 47, 72] as const,
};

const GRADES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"];

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function cleanFilename(value: string) {
  return value.trim().replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "patient";
}

function reportId(date: Date) {
  const stamp = [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
    String(date.getHours()).padStart(2, "0"),
    String(date.getMinutes()).padStart(2, "0"),
    String(date.getSeconds()).padStart(2, "0"),
  ].join("");
  return `RG-${stamp}`;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the retinal image."));
    reader.readAsDataURL(file);
  });
}

function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not prepare an image for the report."));
    image.src = source;
  });
}

async function addContainedImage(
  doc: JsPDF,
  source: string,
  format: "JPEG" | "PNG",
  x: number,
  y: number,
  width: number,
  height: number,
) {
  const image = await loadImage(source);
  const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  const renderedWidth = image.naturalWidth * scale;
  const renderedHeight = image.naturalHeight * scale;
  doc.setFillColor(10, 10, 10);
  doc.rect(x, y, width, height, "F");
  doc.addImage(
    source,
    format,
    x + (width - renderedWidth) / 2,
    y + (height - renderedHeight) / 2,
    renderedWidth,
    renderedHeight,
    undefined,
    "FAST",
  );
}

function addHeader(doc: JsPDF, title: string, subtitle: string, id: string, date: Date) {
  doc.setFillColor(...COLORS.ink);
  doc.rect(0, 0, 210, 31, "F");
  doc.setFillColor(...COLORS.orange);
  doc.rect(0, 0, 7, 31, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(17);
  doc.text("RetinaGrade", 15, 13);
  doc.setFontSize(8.5);
  doc.setFont("helvetica", "normal");
  doc.text(title, 15, 20);
  doc.setTextColor(198, 203, 207);
  doc.text(subtitle, 15, 26);
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(8);
  doc.text(id, 195, 13, { align: "right" });
  doc.text(date.toLocaleString(), 195, 20, { align: "right" });
}

function addFooter(doc: JsPDF, label: string) {
  const pages = doc.getNumberOfPages();
  for (let page = 1; page <= pages; page += 1) {
    doc.setPage(page);
    doc.setDrawColor(...COLORS.line);
    doc.line(15, 282, 195, 282);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(...COLORS.muted);
    doc.text(label, 15, 288);
    doc.text(`Page ${page} of ${pages}`, 195, 288, { align: "right" });
  }
}

function sectionTitle(doc: JsPDF, title: string, y: number) {
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(...COLORS.ink);
  doc.text(title.toUpperCase(), 15, y);
  doc.setDrawColor(...COLORS.line);
  doc.line(15, y + 3, 195, y + 3);
}

function keyValue(doc: JsPDF, label: string, value: string, x: number, y: number, width = 50) {
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(...COLORS.muted);
  doc.text(label.toUpperCase(), x, y);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9.5);
  doc.setTextColor(...COLORS.ink);
  const lines = doc.splitTextToSize(value || "Not provided", width);
  doc.text(lines, x, y + 5);
}

function addProbabilityChart(doc: JsPDF, prediction: BrowserPrediction, y: number) {
  prediction.probabilities.forEach((probability, grade) => {
    const rowY = y + grade * 9;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(...COLORS.ink);
    doc.text(`${grade}  ${GRADES[grade]}`, 15, rowY + 4);
    doc.setFillColor(232, 234, 235);
    doc.rect(66, rowY, 104, 5, "F");
    const barColor = grade === prediction.grade ? COLORS.orange : COLORS.teal;
    doc.setFillColor(barColor[0], barColor[1], barColor[2]);
    doc.rect(66, rowY, Math.max(1, 104 * probability), 5, "F");
    doc.setFont("helvetica", "bold");
    doc.text(percent(probability), 195, rowY + 4, { align: "right" });
  });
}

function qualitySummary(prediction: BrowserPrediction) {
  const quality = prediction.quality;
  const warning = quality.warnings.length ? quality.warnings.join(" ") : "No automated quality warnings.";
  return `${quality.width} x ${quality.height}px | Brightness ${quality.brightness.toFixed(1)} | Sharpness ${quality.sharpness.toFixed(1)} | Visible area ${percent(quality.visible_area_ratio)}. ${warning}`;
}

function plainLanguageGrade(grade: number) {
  if (grade === 0) {
    return "The research model did not favor a diabetic retinopathy grade in this image. This does not rule out eye disease or replace a complete eye examination.";
  }
  if (grade === 1) {
    return "The research model favored the mild diabetic retinopathy category. A qualified eye-care professional must confirm what this means for you.";
  }
  if (grade === 2) {
    return "The research model favored the moderate diabetic retinopathy category. Your eye-care professional should interpret the image and decide the appropriate follow-up.";
  }
  return "The research model favored an advanced diabetic retinopathy category. Only your eye-care professional can confirm the finding and determine timing or treatment.";
}

export async function generateDoctorReport({ file, prediction, details }: ReportContext) {
  const [{ jsPDF }, originalImage] = await Promise.all([import("jspdf"), fileToDataUrl(file)]);
  const doc = new jsPDF({ unit: "mm", format: "a4", compress: true });
  const createdAt = new Date();
  const id = reportId(createdAt);
  const imageFormat = file.type === "image/png" ? "PNG" : "JPEG";
  const referable = prediction.probabilities.slice(2).reduce((sum, value) => sum + value, 0);
  const expectedGrade = prediction.probabilities.reduce(
    (sum, probability, grade) => sum + probability * grade,
    0,
  );

  addHeader(
    doc,
    "Clinician Review Report",
    "AI-assisted research output - clinician interpretation required",
    id,
    createdAt,
  );
  sectionTitle(doc, "Case information", 42);
  keyValue(doc, "Patient", details.patientName, 15, 50, 42);
  keyValue(doc, "Patient ID", details.patientId, 60, 50, 32);
  keyValue(doc, "Reviewing clinician", details.clinicianName, 100, 50, 42);
  keyValue(doc, "Clinic", details.clinicName, 150, 50, 45);

  sectionTitle(doc, "Model assessment", 69);
  doc.setFillColor(...COLORS.paper);
  doc.roundedRect(15, 76, 180, 28, 2, 2, "F");
  keyValue(doc, "Predicted grade", `Grade ${prediction.grade} - ${GRADES[prediction.grade]}`, 21, 84, 48);
  keyValue(doc, "Confidence", percent(prediction.confidence), 77, 84, 25);
  keyValue(doc, "Referable DR likelihood", percent(referable), 112, 84, 31);
  keyValue(doc, "Expected grade", expectedGrade.toFixed(2), 158, 84, 26);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(...COLORS.red);
  doc.text("These values describe model output, not a diagnosis, referral decision, or treatment recommendation.", 21, 99);

  sectionTitle(doc, "Retinal image and model focus", 116);
  await addContainedImage(doc, originalImage, imageFormat, 15, 123, 82, 64);
  await addContainedImage(doc, prediction.explanation.image_url, "JPEG", 113, 123, 82, 64);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(...COLORS.muted);
  doc.text("Preprocessed retinal image", 56, 192, { align: "center" });
  doc.text("Class activation map", 154, 192, { align: "center" });
  doc.setFontSize(7);
  doc.text("Warm areas influenced the selected class; they are not lesion locations.", 154, 197, { align: "center" });

  sectionTitle(doc, "Class probabilities", 210);
  addProbabilityChart(doc, prediction, 217);

  doc.addPage();
  addHeader(doc, "Clinician Review Report", "Detailed model context and sign-off", id, createdAt);
  sectionTitle(doc, "Ordinal thresholds", 42);
  prediction.ordinal_probabilities.forEach((probability, threshold) => {
    keyValue(doc, `Probability grade > ${threshold}`, percent(probability), 15 + threshold * 45, 50, 38);
  });

  sectionTitle(doc, "Image quality", 72);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.ink);
  doc.text(doc.splitTextToSize(qualitySummary(prediction), 180), 15, 80);

  sectionTitle(doc, "Clinician notes", 103);
  doc.setFillColor(...COLORS.paper);
  doc.roundedRect(15, 110, 180, 58, 2, 2, "F");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.ink);
  doc.text(
    doc.splitTextToSize(details.clinicianNotes.trim() || "No clinician notes were entered.", 168).slice(0, 18),
    21,
    119,
  );

  sectionTitle(doc, "Patient communication plan", 182);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.text(
    doc.splitTextToSize(details.patientNextSteps.trim() || "No patient-facing next steps were entered.", 180).slice(0, 10),
    15,
    190,
  );

  doc.setDrawColor(...COLORS.line);
  doc.line(15, 239, 88, 239);
  doc.line(112, 239, 195, 239);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(...COLORS.muted);
  doc.text("CLINICIAN SIGNATURE", 15, 245);
  doc.text("REVIEW DATE", 112, 245);

  doc.setFillColor(255, 246, 241);
  doc.roundedRect(15, 260, 180, 16, 2, 2, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(7.5);
  doc.setTextColor(...COLORS.red);
  doc.text("LIMITATION", 20, 267);
  doc.setFont("helvetica", "normal");
  doc.text(
    "RetinaGrade is a research demonstration and is not cleared as a medical device. Verify all findings independently.",
    20,
    272,
  );

  addFooter(doc, "Confidential clinician review draft - handle according to local privacy requirements.");
  doc.save(`RetinaGrade-clinician-${cleanFilename(details.patientName)}-${id}.pdf`);
}

export async function generatePatientReport({ file, prediction, details }: ReportContext) {
  const [{ jsPDF }, originalImage] = await Promise.all([import("jspdf"), fileToDataUrl(file)]);
  const doc = new jsPDF({ unit: "mm", format: "a4", compress: true });
  const createdAt = new Date();
  const id = reportId(createdAt);
  const imageFormat = file.type === "image/png" ? "PNG" : "JPEG";

  addHeader(
    doc,
    "Patient Information Summary",
    "Prepared for patient communication after clinician review",
    id,
    createdAt,
  );
  sectionTitle(doc, "For you", 43);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(...COLORS.ink);
  const patientNameWidth = doc.getTextWidth(details.patientName);
  if (patientNameWidth > 180) {
    doc.setFontSize(Math.max(12, (18 * 180) / patientNameWidth));
  }
  doc.text(details.patientName, 15, 55);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.muted);
  doc.text(
    `Reviewed by ${details.clinicianName}${details.clinicName ? ` at ${details.clinicName}` : ""}`,
    15,
    62,
  );

  doc.setFillColor(...COLORS.paper);
  doc.roundedRect(15, 72, 180, 42, 2, 2, "F");
  doc.setFillColor(...COLORS.orange);
  doc.rect(15, 72, 5, 42, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(...COLORS.muted);
  doc.text("RESEARCH MODEL CATEGORY", 27, 83);
  doc.setFontSize(19);
  doc.setTextColor(...COLORS.ink);
  doc.text(`Grade ${prediction.grade} - ${GRADES[prediction.grade]}`, 27, 94);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.text(`Model confidence: ${percent(prediction.confidence)}`, 27, 103);
  doc.setFontSize(7.5);
  doc.setTextColor(...COLORS.red);
  doc.text("This category is not, by itself, a medical diagnosis.", 27, 109);

  sectionTitle(doc, "What this means", 129);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10.5);
  doc.setTextColor(...COLORS.ink);
  doc.text(doc.splitTextToSize(plainLanguageGrade(prediction.grade), 180), 15, 139);

  sectionTitle(doc, "Your images", 166);
  await addContainedImage(doc, originalImage, imageFormat, 15, 173, 82, 62);
  await addContainedImage(doc, prediction.explanation.image_url, "JPEG", 113, 173, 82, 62);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(...COLORS.muted);
  doc.text("Retinal image used for review", 56, 240, { align: "center" });
  doc.text("Model focus map", 154, 240, { align: "center" });
  doc.setFontSize(7);
  doc.text("Color shows model influence, not disease locations.", 154, 245, { align: "center" });

  doc.addPage();
  addHeader(doc, "Patient Information Summary", "Clinician instructions and safety information", id, createdAt);
  sectionTitle(doc, "What your clinician recommends next", 43);
  doc.setFillColor(239, 248, 247);
  doc.roundedRect(15, 50, 180, 65, 2, 2, "F");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.ink);
  doc.text(
    doc.splitTextToSize(
      details.patientNextSteps.trim() || "Please follow the instructions discussed with your eye-care professional.",
      166,
    ).slice(0, 17),
    22,
    62,
  );

  sectionTitle(doc, "Important information", 132);
  const reminders = [
    "This summary records a clinician-reviewed research model output; it is not a prescription or a stand-alone diagnosis.",
    "Do not delay or change medical care because of this document.",
    "Contact your eye-care professional if your vision changes or if you have questions about the result.",
    "Keep attending eye examinations at the frequency recommended by your clinician.",
  ];
  reminders.forEach((reminder, index) => {
    const y = 143 + index * 19;
    doc.setFillColor(...COLORS.teal);
    doc.circle(18, y - 1, 1.8, "F");
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(...COLORS.ink);
    doc.text(doc.splitTextToSize(reminder, 168), 24, y);
  });

  sectionTitle(doc, "Contact", 224);
  keyValue(doc, "Clinician", details.clinicianName, 15, 233, 75);
  keyValue(doc, "Clinic", details.clinicName, 105, 233, 75);

  doc.setFillColor(255, 246, 241);
  doc.roundedRect(15, 255, 180, 20, 2, 2, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(...COLORS.red);
  doc.text("RESEARCH USE NOTICE", 20, 263);
  doc.setFont("helvetica", "normal");
  doc.text(
    doc.splitTextToSize("RetinaGrade is not a medical device. Your clinician remains responsible for interpretation and care decisions.", 165),
    20,
    269,
  );

  addFooter(doc, "Patient information summary - discuss questions with your qualified eye-care professional.");
  doc.save(`RetinaGrade-patient-${cleanFilename(details.patientName)}-${id}.pdf`);
}
