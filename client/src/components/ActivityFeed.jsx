import React from 'react';
import { CheckCircle, Info, MessageCircle, FileText, Clock } from 'lucide-react';

const ActivityFeed = ({ activities }) => {
  const getActivityConfig = (type) => {
    const configs = {
      success: { 
        icon: CheckCircle, 
        bg: 'bg-emerald-50', 
        iconColor: 'text-emerald-600',
        border: 'border-emerald-100'
      },
      info: { 
        icon: Info, 
        bg: 'bg-blue-50', 
        iconColor: 'text-blue-600',
        border: 'border-blue-100'
      },
      primary: { 
        icon: MessageCircle, 
        bg: 'bg-indigo-50', 
        iconColor: 'text-indigo-600',
        border: 'border-indigo-100'
      },
      secondary: { 
        icon: FileText, 
        bg: 'bg-slate-50', 
        iconColor: 'text-slate-600',
        border: 'border-slate-100'
      }
    };
    return configs[type] || configs.secondary;
  };

  return (
    <div className="space-y-4">
      {activities.map((activity, index) => {
        const config = getActivityConfig(activity.type);
        const Icon = config.icon;
        return (
          <div 
            key={index} 
            className={`group flex items-start gap-4 p-4 rounded-2xl border ${config.border} ${config.bg} hover:shadow-md transition-all duration-300 cursor-pointer`}
          >
            <div className={`flex-shrink-0 w-10 h-10 rounded-xl ${config.bg} border ${config.border} flex items-center justify-center`}>
              <Icon className={`w-5 h-5 ${config.iconColor}`} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-slate-800 group-hover:text-slate-900 transition-colors">{activity.text}</p>
              <div className="flex items-center gap-1.5 mt-1.5">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                <p className="text-xs text-slate-500">{activity.time}</p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default ActivityFeed;