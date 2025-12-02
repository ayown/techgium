import { useState } from "react";

type Step = 1 | 2 | 3;

export const DiagnosisWizard = () => {
  const [step, setStep] = useState<Step>(1);
  const [symptoms, setSymptoms] = useState("");
  const [labFile, setLabFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const next = () => setStep((s) => (s < 3 ? ((s + 1) as Step) : s));
  const back = () => setStep((s) => (s > 1 ? ((s - 1) as Step) : s));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsProcessing(true);
    // Placeholder: here you'd call your FastAPI backend with text + files.
    setTimeout(() => {
      setIsProcessing(false);
      // Navigate to results route in a real app; for now just show final step.
      setStep(3);
    }, 1500);
  };

  return (
    <form className="card wizard" onSubmit={handleSubmit}>
      <div className="wizard-header">
        <h2>Start Diagnosis</h2>
        <p className="muted">
          Multi-modal intake flow: symptoms, lab reports, and diagnostic images.
        </p>
        <div className="wizard-steps">
          <span className={step >= 1 ? "wizard-step active" : "wizard-step"}>
            1. Symptoms
          </span>
          <span className={step >= 2 ? "wizard-step active" : "wizard-step"}>
            2. Uploads
          </span>
          <span className={step === 3 ? "wizard-step active" : "wizard-step"}>
            3. Processing
          </span>
        </div>
      </div>

      {step === 1 && (
        <section>
          <h3>Describe your symptoms</h3>
          <p className="muted">
            Free-text description will be analyzed by an NER model to extract
            medical entities.
          </p>
          <textarea
            rows={5}
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            placeholder="e.g. My right eye has been red and itchy for 3 days with mild pain..."
          />
        </section>
      )}

      {step === 2 && (
        <section className="uploads-grid">
          <div>
            <h3>Lab Reports (PDF / Images)</h3>
            <p className="muted">
              These will be processed via OCR to extract structured values.
            </p>
            <input
              type="file"
              accept=".pdf,image/*"
              onChange={(e) => setLabFile(e.target.files?.[0] ?? null)}
            />
            {labFile && (
              <p className="file-label">Selected: {labFile.name}</p>
            )}
          </div>

          <div>
            <h3>Diagnostic Imaging (X-ray / Skin / Eye)</h3>
            <p className="muted">
              Upload a clear image. The system may route this to specialized
              imaging models.
            </p>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
            />
            {imageFile && (
              <p className="file-label">Selected: {imageFile.name}</p>
            )}
          </div>
        </section>
      )}

      {step === 3 && (
        <section className="processing-section">
          <h3>Running AI Inference</h3>
          <p className="muted">
            Your text, lab reports, and images are being analyzed in parallel to
            create a unified risk profile.
          </p>
          <div className="loader" />
        </section>
      )}

      <div className="wizard-footer">
        <div>
          {step > 1 && (
            <button
              type="button"
              onClick={back}
              className="secondary-btn"
              disabled={isProcessing}
            >
              Back
            </button>
          )}
        </div>
        <div className="wizard-footer-right">
          {step < 3 && (
            <button
              type="button"
              className="secondary-btn"
              onClick={next}
              disabled={isProcessing}
            >
              Next
            </button>
          )}
          {step === 2 && (
            <button
              type="submit"
              className="primary-btn"
              disabled={isProcessing}
            >
              {isProcessing ? "Processing..." : "Run Diagnosis"}
            </button>
          )}
        </div>
      </div>
    </form>
  );
};


