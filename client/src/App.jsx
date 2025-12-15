import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import Layout from "./components/Layout/Layout";
import Dashboard from "./pages/Dashboard";
import ChatInterface from "./pages/ChatInterface";
import History from "./pages/History";
import Reports from "./pages/Reports";
import Profile from "./pages/Profile";
import Settings from "./pages/Settings";

/**
 * Main App Component
 * - Sets up routing for the entire application
 * - Wraps all pages with the Layout component for consistent UI
 * - Includes toast notifications for user feedback
 */
function App() {
	return (
		<Router>
			{/* Toast notifications for user feedback */}
			<Toaster 
				position="top-right"
				toastOptions={{
					duration: 4000,
					style: {
						background: '#363636',
						color: '#fff',
					},
				}}
			/>
			
			{/* Layout wrapper provides sidebar navigation and header */}
			<Layout>
				<Routes>
					{/* Main dashboard - overview of user's health status */}
					<Route path="/" element={<Dashboard />} />
					
					{/* Smart questionnaire chat interface */}
					<Route path="/chat" element={<ChatInterface />} />
					
					{/* User's diagnostic history */}
					<Route path="/history" element={<History />} />
					
					{/* Generated health reports */}
					<Route path="/reports" element={<Reports />} />
					
					{/* User profile and medical information */}
					<Route path="/profile" element={<Profile />} />
					
					{/* Application settings */}
					<Route path="/settings" element={<Settings />} />
					
					{/* 404 fallback */}
					<Route path="*" element={<div className="flex items-center justify-center h-full"><h1 className="text-2xl text-gray-600">404 - Page Not Found</h1></div>} />
				</Routes>
			</Layout>
		</Router>
	);
}

export default App;
