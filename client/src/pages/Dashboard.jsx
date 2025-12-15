import React from 'react';

/**
 * Dashboard Component
 * - Main overview page showing user's health status
 * - Quick access to key features and recent activity
 * - Health metrics and summary cards
 */
const Dashboard = () => {
  return (
    <div className="space-y-8">
      {/* Dashboard Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Health Dashboard</h1>
          <p className="mt-1 text-slate-500">Welcome back! Here's your daily health overview.</p>
        </div>
        <button className="inline-flex items-center justify-center px-4 py-2 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors">
          Start New Assessment
        </button>
      </div>

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Metric Card 1 */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider">Assessments</h3>
            <div className="p-2 bg-blue-50 rounded-lg">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
          </div>
          <div className="flex items-baseline mb-2">
            <span className="text-3xl font-bold text-slate-900">12</span>
            <span className="ml-2 text-sm font-medium text-emerald-600 flex items-center">
              <svg className="w-4 h-4 mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              +2 this week
            </span>
          </div>
          <div className="w-full bg-slate-100 rounded-full h-1.5 mt-2">
            <div className="bg-blue-600 h-1.5 rounded-full" style={{ width: '70%' }}></div>
          </div>
        </div>

        {/* Metric Card 2 */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider">Last Checkup</h3>
            <div className="p-2 bg-emerald-50 rounded-lg">
              <svg className="w-6 h-6 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <div className="flex items-baseline mb-1">
            <span className="text-3xl font-bold text-slate-900">2 days</span>
            <span className="ml-1 text-lg text-slate-500">ago</span>
          </div>
          <p className="text-sm text-slate-600">General Checkup</p>
        </div>

        {/* Metric Card 3 */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wider">Health Score</h3>
            <div className="p-2 bg-amber-50 rounded-lg">
              <svg className="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
          </div>
          <div className="flex items-baseline mb-2">
            <span className="text-3xl font-bold text-slate-900">85</span>
            <span className="ml-1 text-sm text-slate-400">/100</span>
          </div>
          <div className="w-full bg-slate-100 rounded-full h-1.5 mt-2">
             <div className="bg-amber-500 h-1.5 rounded-full" style={{ width: '85%' }}></div>
          </div>
        </div>
      </div>

      {/* Main Content Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Recent Activity - 2 Cols */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between">
            <h2 className="text-lg font-bold text-slate-900">Recent Activity</h2>
            <button className="text-sm font-medium text-blue-600 hover:text-blue-700">View All</button>
          </div>
          <div className="divide-y divide-slate-100">
            {[1, 2, 3].map((item, i) => (
              <div key={i} className="px-6 py-4 hover:bg-slate-50 transition-colors">
                <div className="flex items-start">
                  <div className={`mt-1 h-2.5 w-2.5 rounded-full flex-shrink-0 ${
                    i === 0 ? 'bg-blue-500' : i === 1 ? 'bg-emerald-500' : 'bg-amber-500'
                  }`}></div>
                  <div className="ml-4 flex-1">
                    <p className="text-sm font-medium text-slate-900">
                      {i === 0 ? 'Completed health assessment' : i === 1 ? 'Uploaded lab report' : 'Started new chat session'}
                    </p>
                    <div className="mt-1 flex items-center text-xs text-slate-500 space-x-2">
                      <span>{i === 0 ? '2 days ago' : i === 1 ? '5 days ago' : '1 week ago'}</span>
                      <span>•</span>
                      <span>{i === 0 ? 'General Checkup' : i === 1 ? 'Blood Work' : 'Consultation'}</span>
                    </div>
                  </div>
                  <button className="text-slate-400 hover:text-blue-600">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions - 1 Col */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm">
          <div className="px-6 py-5 border-b border-slate-100">
            <h2 className="text-lg font-bold text-slate-900">Quick Actions</h2>
          </div>
          <div className="p-6 space-y-3">
             <button className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-blue-50 border border-slate-100 hover:border-blue-100 rounded-xl transition-all group">
                <div className="flex items-center space-x-3">
                   <div className="p-2 bg-white rounded-lg shadow-sm group-hover:bg-blue-600 transition-colors">
                      <svg className="w-5 h-5 text-blue-600 group-hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                      </svg>
                   </div>
                   <span className="font-medium text-slate-700 group-hover:text-blue-700">Start Health Chat</span>
                </div>
                <svg className="w-5 h-5 text-slate-400 group-hover:text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
             </button>
             
             <button className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-emerald-50 border border-slate-100 hover:border-emerald-100 rounded-xl transition-all group">
                <div className="flex items-center space-x-3">
                   <div className="p-2 bg-white rounded-lg shadow-sm group-hover:bg-emerald-600 transition-colors">
                      <svg className="w-5 h-5 text-emerald-600 group-hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                   </div>
                   <span className="font-medium text-slate-700 group-hover:text-emerald-700">Upload Lab Report</span>
                </div>
                <svg className="w-5 h-5 text-slate-400 group-hover:text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
             </button>

             <button className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-purple-50 border border-slate-100 hover:border-purple-100 rounded-xl transition-all group">
                <div className="flex items-center space-x-3">
                   <div className="p-2 bg-white rounded-lg shadow-sm group-hover:bg-purple-600 transition-colors">
                      <svg className="w-5 h-5 text-purple-600 group-hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                   </div>
                   <span className="font-medium text-slate-700 group-hover:text-purple-700">View Reports</span>
                </div>
                <svg className="w-5 h-5 text-slate-400 group-hover:text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
             </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;