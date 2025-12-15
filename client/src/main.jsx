import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import React from "react";
/**
 * Main entry point for the Chiranjeevi Health Assistant application
 * - Sets up React root with StrictMode for development warnings
 * - App component already includes BrowserRouter, so no need to wrap here
 */
createRoot(document.getElementById("root")).render(
	<>
		<App />
	</>
);
