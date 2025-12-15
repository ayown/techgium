import React from 'react';

/**
 * Header Component
 * - Top navigation bar with mobile menu button
 * - Search functionality and user actions
 * - Responsive design for mobile and desktop
 * 
 * @param {Object} props - Component props
 * @param {Function} props.onMenuClick - Callback to toggle mobile sidebar
 */
const Header = ({ onMenuClick }) => {
  return (
    <header className="sticky top-0 z-20 h-20 bg-white border-b border-slate-200 px-6 lg:px-10 flex items-center justify-between">
      
      {/* Left section - Mobile menu button and search */}
      <div className="flex items-center space-x-4 flex-1">
        {/* Mobile menu button - only visible on mobile */}
        <button
          onClick={onMenuClick}
          className="lg:hidden p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          aria-label="Open sidebar"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        {/* Search bar - refined professional look */}
        <div className="hidden sm:block relative max-w-md w-full">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <svg className="h-5 w-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input
            type="text"
            placeholder="Search health records..."
            className="block w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg leading-5 bg-slate-50 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 focus:bg-white transition-all duration-200 sm:text-sm"
          />
        </div>
      </div>

      {/* Right section - Notifications and user menu */}
      <div className="flex items-center space-x-3 lg:space-x-6">
        
        {/* Emergency button - cleaner look */}
        <button className="hidden sm:flex items-center px-4 py-2 border border-red-200 text-red-600 bg-red-50 rounded-lg hover:bg-red-100 hover:border-red-300 focus:outline-none focus:ring-2 focus:ring-red-500/20 transition-all duration-200">
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <span className="text-sm font-semibold">Emergency</span>
        </button>

        {/* Notifications */}
        <button className="relative p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-full transition-all">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5-5V9a6 6 0 10-12 0v3l-5 5h5m7 0v1a3 3 0 01-6 0v-1m6 0H9" />
          </svg>
          {/* Notification badge */}
          <span className="absolute top-2 right-2 block h-2.5 w-2.5 rounded-full bg-red-500 border-2 border-white"></span>
        </button>

        {/* Separator */}
        <div className="h-6 w-px bg-slate-200 hidden sm:block"></div>

        {/* User profile styling */}
        <div className="flex items-center space-x-3 cursor-pointer">
           <div className="hidden md:block text-right">
             <p className="text-sm font-medium text-slate-700">John Doe</p>
             <p className="text-xs text-slate-500">Patient</p>
           </div>
           <div className="w-9 h-9 bg-slate-200 rounded-full flex items-center justify-center border border-slate-100">
              <span className="text-slate-600 text-sm font-semibold">JD</span>
           </div>
        </div>
      </div>
    </header>
  );
};

export default Header;