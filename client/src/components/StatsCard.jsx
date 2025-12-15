import React from 'react';
import { TrendingUp } from 'lucide-react';

const StatsCard = ({ title, value, color, icon: Icon }) => {
  const colorConfig = {
    blue: {
      gradient: 'from-blue-500 to-indigo-600',
      glow: 'group-hover:shadow-blue-500/25',
      iconBg: 'bg-blue-500/10',
      iconColor: 'text-blue-600',
      accent: 'text-blue-600'
    },
    green: {
      gradient: 'from-emerald-500 to-teal-600',
      glow: 'group-hover:shadow-emerald-500/25',
      iconBg: 'bg-emerald-500/10',
      iconColor: 'text-emerald-600',
      accent: 'text-emerald-600'
    },
    yellow: {
      gradient: 'from-amber-500 to-orange-600',
      glow: 'group-hover:shadow-amber-500/25',
      iconBg: 'bg-amber-500/10',
      iconColor: 'text-amber-600',
      accent: 'text-amber-600'
    },
    red: {
      gradient: 'from-rose-500 to-pink-600',
      glow: 'group-hover:shadow-rose-500/25',
      iconBg: 'bg-rose-500/10',
      iconColor: 'text-rose-600',
      accent: 'text-rose-600'
    }
  };

  const config = colorConfig[color] || colorConfig.blue;

  return (
    <div className={`group relative bg-white rounded-2xl p-6 border border-slate-100 shadow-sm hover:shadow-2xl ${config.glow} transition-all duration-500 hover:-translate-y-1 overflow-hidden`}>
      {/* Subtle gradient accent line at top */}
      <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${config.gradient} opacity-80`}></div>
      
      {/* Background glow effect */}
      <div className={`absolute -top-12 -right-12 w-32 h-32 bg-gradient-to-br ${config.gradient} opacity-5 rounded-full blur-2xl group-hover:opacity-10 transition-opacity duration-500`}></div>
      
      <div className="relative flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-2">{title}</p>
          <p className="text-3xl font-bold text-slate-900 font-serif tracking-tight">
            {value}
          </p>
          <div className="flex items-center gap-1.5 mt-3">
            <TrendingUp className={`w-4 h-4 ${config.accent}`} />
            <span className={`text-sm font-semibold ${config.accent}`}>+12%</span>
            <span className="text-xs text-slate-400">vs last week</span>
          </div>
        </div>
        <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${config.iconBg} ring-1 ring-slate-900/5`}>
          <Icon className={`w-7 h-7 ${config.iconColor}`} />
        </div>
      </div>
    </div>
  );
};

export default StatsCard;