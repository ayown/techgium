
'use client';

import React, { useState } from 'react';
import { MessageBubble } from '@/components/chat/message-bubble';
import { ChatInput } from '@/components/chat/chat-input';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatbotPage() {
  const [input, setInput] = useState('');
  
  // Dummy Data
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm **CHIRANJEEVAI**, your medical assistant. How can I help you today?"
    },
    {
      id: '2',
      role: 'user',
      content: "I have a headache and mild fever."
    },
    {
      id: '3',
      role: 'assistant',
      content: "I understand. To better assess your condition, could you tell me:\n\n1. How long have you had the fever?\n2. Is the headache localized to one area?\n3. Any other symptoms like nausea or sensitivity to light?"
    }
  ]);

  const handleSend = () => {
    if (!input.trim()) return;

    // Add user message
    const newUserMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input
    };

    setMessages(prev => [...prev, newUserMsg]);
    setInput('');

    // Simulate fake response
    setTimeout(() => {
      const newAssistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "This is a **dummy response** for UI testing. Backend integration is coming in Phase 3."
      };
      setMessages(prev => [...prev, newAssistantMsg]);
    }, 1000);
  };

  return (
    <div className="chatbot-container flex flex-col h-full bg-surface overflow-hidden">
      {/* Header */}
      <div className="chat-header p-[24px] bg-surface-container flex justify-between items-center shrink-0">
        <div>
            <h1 className="chat-title text-[24px] font-medium text-on-surface leading-tight">CHIRANJEEVAI</h1>
            <p className="subtitle text-[16px] text-on-surface-variant mt-[8px]">AI Medical Assistant</p>
        </div>
        <div className="header-controls flex gap-[12px] items-center">
            <div className="connection-status px-[16px] py-[8px] bg-success-container text-on-success-container rounded-full text-[12px] font-medium">
                Connected
            </div>
        </div>
      </div>

      {/* Chat Display */}
      <div className="chat-display flex-1 p-[24px] overflow-y-auto flex flex-col gap-[16px] bg-surface scroll-smooth">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} role={msg.role} content={msg.content} />
        ))}
      </div>

      {/* Input Area */}
      <ChatInput 
        value={input} 
        onChange={setInput} 
        onSend={handleSend} 
      />
    </div>
  );
}
