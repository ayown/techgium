export const Header = () => {
  // In a real app this would use the logged-in user's name and time of day
  const greeting = "Good Morning, Ritesh";
  const subtitle = "Your personalized multi-modal health companion.";

  return (
    <header className="app-header">
      <div>
        <h1 className="app-greeting">{greeting}</h1>
        <p className="app-subtitle">{subtitle}</p>
      </div>
      <div className="app-header-right">
        <button className="primary-btn header-cta">Start Diagnosis</button>
      </div>
    </header>
  );
};


