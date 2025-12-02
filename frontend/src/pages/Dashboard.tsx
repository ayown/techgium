import { useState } from "react";

const dummyVitals = {
  ecg: "Normal sinus rhythm",
  bpm: 76,
  spo2: 98
};

export const Dashboard = () => {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<string[]>([
    "Hi, I'm your AI assistant. Tell me how you're feeling today."
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;
    // For now we just echo back; later this can call your backend chatbot.
    setChat((prev) => [...prev, `You: ${message}`]);
    setMessage("");
  };

  return (
    <div className="grid-2">
      <section className="card">
        <h2>How are you feeling?</h2>
        <p className="muted">
          Describe your symptoms in your own words. The system will use this to
          activate the right diagnostic models.
        </p>

        <div className="chat-window">
          {chat.map((line, idx) => (
            <div key={idx} className="chat-line">
              {line}
            </div>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="chat-input-row">
          <input
            type="text"
            placeholder="e.g. I have been getting headaches and blurred vision..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />
          <button type="submit" className="primary-btn">
            Send
          </button>
        </form>
      </section>

      <section className="card">
        <h2>Vital Signs</h2>
        <p className="muted">Sample data for preview. Connect to devices later.</p>
        <div className="vitals-grid">
          <div className="vital-pill">
            <span className="label">ECG</span>
            <span className="value">{dummyVitals.ecg}</span>
          </div>
          <div className="vital-pill">
            <span className="label">BPM</span>
            <span className="value">{dummyVitals.bpm} bpm</span>
          </div>
          <div className="vital-pill">
            <span className="label">SpO₂</span>
            <span className="value">{dummyVitals.spo2}%</span>
          </div>
        </div>
      </section>
    </div>
  );
};


