import React from 'react';
import { ChevronRight } from 'lucide-react';

const QuickActions = ({ actions }) => {
  return (
    <div className="space-y-3">
      {actions.map((action, index) => {
        const Icon = action.icon;
        return (
          <button 
            key={index} 
            className="group w-full flex items-center justify-between p-4 bg-slate-50/50 hover:bg-white border border-slate-100 rounded-2xl hover:border-indigo-200 hover:shadow-lg hover:shadow-indigo-500/5 transition-all duration-300 hover:-translate-y-0.5"
          >
            <div className="flex items-center gap-4">
              <div className="w-11 h-11 rounded-xl bg-white flex items-center justify-center shadow-sm group-hover:bg-indigo-50 group-hover:shadow-md transition-all border border-slate-100 group-hover:border-indigo-100">
                <Icon className="w-5 h-5 text-slate-500 group-hover:text-indigo-600 transition-colors" />
              </div>
              <div className="text-left">
                <p className="text-sm font-semibold text-slate-800 group-hover:text-indigo-600 transition-colors">
                  {action.name}
                </p>
                <p className="text-xs text-slate-400 mt-0.5">{action.description}</p>
              </div>
            </div>
            <ChevronRight className="w-5 h-5 text-slate-300 group-hover:text-indigo-500 group-hover:translate-x-1 transition-all" />
          </button>
        );
      })}
    </div>
  );
};

export default QuickActions;