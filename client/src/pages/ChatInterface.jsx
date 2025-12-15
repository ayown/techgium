import React, { useState, useRef, useEffect } from 'react';

/**
 * ChatInterface Component
 * - Main smart questionnaire chat interface
 * - Handles conversation flow with AI health assistant
 * - Supports file uploads and multimodal interactions
 * - Real-time messaging with typing indicators
 */
const ChatInterface = () => {
  // Chat state management
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: "Hello! I'm your AI health assistant. I'll help you through a comprehensive health assessment. Let's start with some basic questions. How are you feeling today?",
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  
  // Refs for auto-scrolling and file input
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  /**
   * Auto-scroll to bottom when new messages arrive
   */
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  /**
   * Handle sending a new message
   */
  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim() && selectedFiles.length === 0) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputMessage,
      files: selectedFiles,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setSelectedFiles([]);
    setIsTyping(true);

    // Simulate AI response (replace with actual API call)
    setTimeout(() => {
      const aiResponse = {
        id: Date.now() + 1,
        role: 'assistant',
        content: generateAIResponse(inputMessage),
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
    }, 1500);
  };

  /**
   * Generate AI response based on user input
   * TODO: Replace with actual AI model integration
   */
  const generateAIResponse = (userInput) => {
    const responses = [
      "Thank you for sharing that information. Can you tell me more about when these symptoms started?",
      "I understand. Let me ask you a few more questions to better assess your condition. Do you have any allergies?",
      "Based on what you've told me, I'd like to gather some additional information. Have you experienced this before?",
      "That's helpful information. Can you describe the intensity of your symptoms on a scale of 1-10?",
      "I see. Let's also discuss your medical history. Are you currently taking any medications?"
    ];
    return responses[Math.floor(Math.random() * responses.length)];
  };

  /**
   * Handle file selection
   */
  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(prev => [...prev, ...files]);
  };

  /**
   * Remove selected file
   */
  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  /**
   * Quick response buttons for common answers
   */
  const quickResponses = [
    "Yes", "No", "Not sure", "Tell me more", "I need help"
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
      
      {/* Chat header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-white z-10">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center shadow-sm">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <span className="absolute bottom-0 right-0 w-2.5 h-2.5 bg-emerald-500 border-2 border-white rounded-full"></span>
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-900">AI Health Assistant</h2>
            <p className="text-xs text-slate-500 font-medium flex items-center">
              {isTyping ? (
                <span className="text-blue-600 font-semibold">Typing...</span>
              ) : (
                <span className="text-emerald-600">Online</span>
              )}
            </p>
          </div>
        </div>
        
        {/* Chat actions */}
        <div className="flex items-center space-x-2">
          <button className="p-2 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-50 transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>
      </div>

      {/* Messages container */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth bg-slate-50/50">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} group`}
          >
            {message.role === 'assistant' && (
               <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 text-xs font-bold mr-3 mt-1 flex-shrink-0">
                 AI
               </div>
            )}
            <div className={`max-w-[85%] lg:max-w-xl px-5 py-3 shadow-sm ${
              message.role === 'user'
                ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm'
                : 'bg-white text-slate-800 rounded-2xl rounded-tl-sm border border-slate-200'
            }`}>
              <p className="leading-relaxed text-sm">{message.content}</p>
              
              {/* Display uploaded files */}
              {message.files && message.files.length > 0 && (
                <div className="mt-3 space-y-2">
                  {message.files.map((file, index) => (
                    <div key={index} className={`flex items-center space-x-2 text-xs p-2 rounded-lg ${
                      message.role === 'user' ? 'bg-white/10 text-white' : 'bg-slate-50 text-slate-600'
                    }`}>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                      </svg>
                      <span className="font-medium truncate max-w-[150px]">{file.name}</span>
                    </div>
                  ))}
                </div>
              )}
              
              <p className={`text-[10px] mt-1.5 font-medium ${
                message.role === 'user' ? 'text-blue-100' : 'text-slate-400'
              }`}>
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </p>
            </div>
          </div>
        ))}
        
        {/* Typing indicator */}
        {isTyping && (
          <div className="flex justify-start items-center space-x-3">
             <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 text-xs font-bold">AI</div>
             <div className="bg-white rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm border border-slate-200">
              <div className="flex space-x-1">
                <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></div>
                <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Quick responses */}
      <div className="px-6 py-3 border-t border-slate-100 bg-white">
        <div className="flex flex-wrap gap-2">
          {quickResponses.map((response, index) => (
            <button
              key={index}
              onClick={() => setInputMessage(response)}
              className="px-3 py-1.5 text-xs font-medium bg-slate-50 border border-slate-200 text-slate-600 rounded-full hover:bg-blue-50 hover:border-blue-200 hover:text-blue-600 transition-colors"
            >
              {response}
            </button>
          ))}
        </div>
      </div>

      {/* Selected files display */}
      {selectedFiles.length > 0 && (
        <div className="px-6 py-2 bg-slate-50 border-t border-slate-100">
          <div className="flex flex-wrap gap-2">
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex items-center space-x-2 bg-white border border-blue-100 text-blue-700 px-3 py-1 rounded-lg text-sm shadow-sm">
                <span className="truncate max-w-[200px]">{file.name}</span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-slate-400 hover:text-red-500 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Message input */}
      <div className="p-4 bg-white border-t border-slate-100">
        <form onSubmit={handleSendMessage} className="relative flex items-end space-x-2 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2 focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent transition-all">
          
          {/* File upload button */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex-shrink-0 p-2 text-slate-400 hover:text-blue-600 rounded-lg hover:bg-blue-50 transition-colors"
            title="Attach file"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
            </svg>
          </button>
          
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.pdf,.doc,.docx"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          {/* Text input */}
          <div className="flex-1 py-2">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              className="w-full bg-transparent border-0 focus:ring-0 p-0 text-slate-800 placeholder-slate-400 resize-none max-h-32 text-sm leading-relaxed"
              rows="1"
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
            />
          </div>
          
          {/* Send button */}
          <button
            type="submit"
            disabled={!inputMessage.trim() && selectedFiles.length === 0}
            className="flex-shrink-0 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;