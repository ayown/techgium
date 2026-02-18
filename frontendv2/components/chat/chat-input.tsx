
import React, { useRef, useEffect } from 'react';
import { Send, Paperclip } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
}

export function ChatInput({ value, onChange, onSend, disabled }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="chat-input-area p-[24px] bg-surface-container flex gap-[12px] items-center shrink-0">
      <div className="input-wrapper flex-1 bg-surface-container-high rounded-[12px] flex items-center px-[8px] focus-within:outline focus-within:outline-2 focus-within:outline-primary focus-within:outline-offset-[-2px]">
        <button className="attach-btn p-[12px] bg-transparent border-none cursor-pointer text-[20px] text-on-surface-variant hover:text-primary transition-colors">
          📎
        </button>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={disabled}
          className="chat-input flex-1 p-[16px] pl-[4px] border-none bg-transparent font-inherit text-[16px] text-on-surface resize-none max-h-[120px] focus:ring-0"
          rows={1}
        />
      </div>

      <button 
        onClick={onSend} 
        disabled={!value.trim() || disabled}
        className="send-btn w-[48px] h-[48px] rounded-full border-none bg-primary text-on-primary cursor-pointer flex items-center justify-center text-[20px] transition-all hover:shadow-[0_2px_8px_rgba(11,87,208,0.3)] disabled:bg-surface-container-high disabled:text-on-surface-variant disabled:cursor-not-allowed"
      >
        ➤
      </button>
    </div>
  );
}
