
'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { Home, Stethoscope, Bot, Menu, X, Plus } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

export function Sidebar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  // Toggle for mobile
  const toggleOpen = () => setIsOpen(!isOpen);

  const navItems = [
    { href: '/', label: 'Home', icon: '🏠' },
    { href: '/screening', label: 'Screening', icon: '🩺' },
    { href: '/chatbot', label: 'CHIRANJEEVAI', icon: '🤖' },
  ];

  return (
    <>
      {/* Mobile Menu Button */}
      <button 
        onClick={toggleOpen}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-surface-container-low rounded-full shadow-md"
      >
        {isOpen ? <X className="w-6 h-6 text-on-surface" /> : <Menu className="w-6 h-6 text-on-surface" />}
      </button>

      {/* Sidebar Container */}
      <aside className={cn(
        "fixed lg:sticky top-0 left-0 h-screen bg-surface-container-low border-r border-surface-container-high transition-all duration-300 ease-in-out z-40 flex flex-col",
        isOpen ? "translate-x-0 w-[var(--sidebar-width)]" : "-translate-x-full lg:translate-x-0 w-[var(--sidebar-width)]"
      )}>
        {/* Logo */}
        <div className="logo-container mb-[32px] px-[12px] mt-[24px]">
          <div className="logo">
            Chiranjeevi
          </div>
        </div>

        {/* Navigation */}
        <nav className="nav-menu flex-1 px-[16px] flex flex-col gap-[8px]">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn("nav-item", isActive && "active")}
                onClick={() => setIsOpen(false)} // Close on mobile click
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-text">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="sidebar-footer p-[16px] px-[12px] border-t border-surface-container-high text-[12px] text-outline text-left">
          Chiranjeevi v2.0
        </div>
      </aside>

      {/* Mobile Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/40 lg:hidden z-30 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />
        )}
      </AnimatePresence>
    </>
  );
}
