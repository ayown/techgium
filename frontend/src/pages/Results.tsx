const mockResults = {
  unifiedRiskScore: 0.72,
  confidence: 0.88,
  probableDiagnosis: "Early-stage diabetic retinopathy (suspected)",
  lifestyle: [
    "Maintain stable blood sugar with regular monitoring.",
    "Schedule an annual comprehensive eye exam.",
    "Follow a balanced, low-glycemic diet and regular exercise."
  ],
  physicianFlags: [
    "Recommend ophthalmologist review within 2–4 weeks.",
    "Flag for teleconsult follow-up if symptoms worsen."
  ]
};

export const Results = () => {
  return (
    <div className="card">
      <h2>Unified Risk Profile</h2>
      <p className="muted">
        Combined view of text, lab, and imaging models. Replace with live API
        output later.
      </p>

      <div className="results-metrics">
        <div className="metric">
          <span className="label">Overall Risk Score</span>
          <span className="value">
            {(mockResults.unifiedRiskScore * 100).toFixed(0)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Model Confidence</span>
          <span className="value">
            {(mockResults.confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="metric">
          <span className="label">Probable Diagnosis</span>
          <span className="value">
            {mockResults.probableDiagnosis}
          </span>
        </div>
      </div>

      <div className="results-sections">
        <section>
          <h3>Actionable Lifestyle Recommendations</h3>
          <ul>
            {mockResults.lifestyle.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </section>

        <section>
          <h3>Physician Flags</h3>
          <ul>
            {mockResults.physicianFlags.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
};


