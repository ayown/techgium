import { Route, Routes, Navigate } from "react-router-dom";
import { LayoutShell } from "./layout/LayoutShell";
import { Dashboard } from "./pages/Dashboard";
import { DiagnosisWizard } from "./pages/DiagnosisWizard";
import { Results } from "./pages/Results";
import { History } from "./pages/History";
import { Resources } from "./pages/Resources";
import { Settings } from "./pages/Settings";

export const App = () => {
  return (
    <LayoutShell>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/diagnosis" element={<DiagnosisWizard />} />
        <Route path="/results" element={<Results />} />
        <Route path="/history" element={<History />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </LayoutShell>
  );
};


