import React from 'react';
import StatsCard from '../components/StatsCard';
import ActivityFeed from '../components/ActivityFeed';
import QuickActions from '../components/QuickActions';
import { Activity, FileText, MessageSquare, Calendar, BarChart2, Heart, Clock, Plus, Brain, Sparkles, ArrowRight, TrendingUp } from 'lucide-react';

const Dashboard = () => {
  const stats = [
    { title: 'Total Assessments', value: '12', color: 'blue', icon: BarChart2 },
    { title: 'Health Score', value: '85/100', color: 'green', icon: Heart },
    { title: 'Last Check', value: '2 days ago', color: 'yellow', icon: Clock },
    { title: 'Pending Reports', value: '3', color: 'red', icon: FileText }
  ];

  const activities = [
    { text: 'Completed comprehensive health assessment', time: '2 hours ago', type: 'success' },
    { text: 'Uploaded lab report for blood work', time: '1 day ago', type: 'info' },
    { text: 'Started new chat session with AI assistant', time: '2 days ago', type: 'primary' },
    { text: 'Downloaded health report PDF', time: '3 days ago', type: 'secondary' }
  ];

  const quickActions = [
    { name: 'Start Health Chat', icon: MessageSquare, description: 'Begin AI-powered health assessment' },
    { name: 'Upload Lab Report', icon: FileText, description: 'Add new medical documents' },
    { name: 'View Reports', icon: Activity, description: 'Access your health analytics' },
    { name: 'Schedule Checkup', icon: Calendar, description: 'Book your next appointment' }
  ];

  return (
    <div className="min-h-screen relative">
      {/* Hero Header Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-600 via-blue-600 to-indigo-700"></div>
        {/* Decorative circles */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-white/5 rounded-full blur-3xl -mr-48 -mt-48"></div>
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-indigo-400/10 rounded-full blur-2xl -ml-32 -mb-32"></div>
        
        <div className="relative max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-16 pb-32">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-8 animate-fadeIn">
            <div className="space-y-4">
              <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm px-4 py-2 rounded-full text-white/90 text-sm font-medium border border-white/20">
                <Sparkles className="w-4 h-4" />
                <span>AI-Powered Health Analytics</span>
              </div>
              <h1 className="text-5xl lg:text-6xl font-bold text-white font-serif tracking-tight leading-tight">
                Welcome back,<br />
                <span className="text-indigo-200">Koustav</span>
              </h1>
              <p className="text-xl text-indigo-100 font-light max-w-lg">
                Your personalized health dashboard is ready. Track your wellness journey with precision.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-4">
              <button className="group relative bg-white text-indigo-600 px-8 py-4 rounded-2xl overflow-hidden transition-all hover:shadow-2xl hover:shadow-white/20 active:scale-95 font-bold text-lg">
                <span className="relative flex items-center justify-center gap-3">
                  <Plus className="w-5 h-5" />
                  <span>New Assessment</span>
                </span>
              </button>
              <button className="group bg-white/10 backdrop-blur-sm text-white px-8 py-4 rounded-2xl border border-white/20 hover:bg-white/20 transition-all font-medium text-lg flex items-center justify-center gap-2">
                <span>View Reports</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - Overlapping Cards */}
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 -mt-20 relative z-10 pb-16 space-y-10">
        
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <div key={index} className="animate-fadeIn" style={{ animationDelay: `${index * 100}ms` }}>
               <StatsCard {...stat} />
            </div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Activity Feed */}
          <div className="lg:col-span-8 animate-fadeIn" style={{ animationDelay: '200ms' }}>
            <div className="bg-white rounded-3xl p-8 lg:p-10 premium-shadow h-full border border-slate-100/80 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-indigo-50 to-transparent rounded-full blur-3xl -mr-48 -mt-48 pointer-events-none opacity-70"></div>
              <div className="flex items-center justify-between mb-8 relative z-10">
                <div>
                  <h2 className="text-2xl lg:text-3xl font-serif text-slate-900">Recent Activity</h2>
                  <p className="text-slate-500 mt-1">Your latest health actions</p>
                </div>
                <button className="group flex items-center gap-2 text-indigo-600 hover:text-indigo-700 font-semibold text-sm tracking-wide uppercase bg-indigo-50 hover:bg-indigo-100 px-4 py-2 rounded-xl transition-all">
                  <span>View All</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                </button>
              </div>
              <ActivityFeed activities={activities} />
            </div>
          </div>
          
          {/* Quick Actions & Insights Column */}
          <div className="lg:col-span-4 space-y-6 animate-fadeIn" style={{ animationDelay: '300ms' }}>
            {/* Quick Actions */}
            <div className="bg-white rounded-3xl p-6 lg:p-8 premium-shadow border border-slate-100/80">
              <h2 className="text-xl lg:text-2xl font-serif text-slate-900 mb-6">Quick Actions</h2>
              <QuickActions actions={quickActions} />
            </div>

            {/* Health Insights - Floating Design */}
            <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6 lg:p-8 shadow-2xl text-white group">
              <div className="absolute top-0 right-0 w-48 h-48 bg-indigo-500/20 rounded-full blur-3xl -mr-16 -mt-16 group-hover:bg-indigo-500/30 transition-all duration-700"></div>
              <div className="absolute bottom-0 left-0 w-32 h-32 bg-blue-500/10 rounded-full blur-2xl -ml-8 -mb-8"></div>
              
              <div className="relative z-10">
                <div className="flex items-center gap-3 mb-5">
                  <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-500/30">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex items-center gap-2 text-indigo-300 text-sm font-medium">
                    <TrendingUp className="w-4 h-4" />
                    <span>Trending Up</span>
                  </div>
                </div>
                <h3 className="text-2xl font-serif mb-3 text-white">AI Insights</h3>
                <p className="text-slate-300 font-light leading-relaxed mb-6 text-sm lg:text-base">
                  Your cardiovascular trends are looking great this week. Keep up the excellent work!
                </p>
                <button className="w-full bg-white/10 hover:bg-white/20 backdrop-blur-sm text-white font-medium py-3 px-6 rounded-xl transition-all border border-white/10 hover:border-white/20 flex items-center justify-center gap-2 group/btn">
                  <span>View Full Report</span>
                  <ArrowRight className="w-4 h-4 group-hover/btn:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
