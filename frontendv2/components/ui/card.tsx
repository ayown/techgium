
import React from 'react';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function Card({ className = '', children, ...props }: CardProps) {
  return (
    <div 
      className={`bg-surface-container p-6 rounded-[28px] transition-colors duration-200 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
