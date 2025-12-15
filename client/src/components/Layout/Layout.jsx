import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';

/**
 * Layout Component
 * - Provides the main application structure with sidebar and header
 * - Handles responsive design for mobile and desktop
 * - Manages sidebar toggle state for mobile devices
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - Page content to render
 */
const Layout = ({ children }) => {
  // State to control sidebar visibility on mobile devices
  const [sidebarOpen, setSidebarOpen] = useState(false);

  /**
   * Toggle sidebar visibility (primarily for mobile)
   */
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  /**
   * Close sidebar (used when clicking outside or on mobile nav items)
   */
  const closeSidebar = () => {
    setSidebarOpen(false);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* Sidebar - Hidden on mobile by default, shown when sidebarOpen is true */}
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={closeSidebar}
      />
      
      {/* Mobile overlay - appears when sidebar is open on mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-20 lg:hidden"
          onClick={closeSidebar}
        />
      )}
      
      {/* Main content area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header with mobile menu button */}
        <Header onMenuClick={toggleSidebar} />
        
        {/* Page content - Centered container with professional spacing */}
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-6 lg:p-10 scroll-smooth">
          <div className="max-w-7xl mx-auto space-y-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;