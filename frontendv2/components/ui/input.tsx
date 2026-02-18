
import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export function Input({ className = '', ...props }: InputProps) {
  return (
    <input 
      className={`w-full h-14 px-4 bg-surface-container-high border-none rounded-xl text-base text-on-surface transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary focus:bg-surface-container-high placeholder:text-on-surface-variant/50 ${className}`}
      {...props}
    />
  );
}
