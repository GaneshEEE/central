import React, { useState, useRef, useEffect } from 'react';
import { Settings, Key, RotateCcw, Check } from 'lucide-react';

interface CircularLauncherProps {
  onClick: () => void;
}

interface ApiKeyOption {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'error';
  configured: boolean;
}

declare global {
  interface Window {
    selectedApiKey?: string;
  }
}

// Dynamically get all VITE_GENAI_API_KEY_* from import.meta.env, sorted by number
function getApiKeyOptions(): ApiKeyOption[] {
  const keys: ApiKeyOption[] = [];
  for (let i = 1; i <= 5; i++) {
    const envKey = `VITE_GENAI_API_KEY_${i}`;
    keys.push({
      id: `GENAI_API_KEY_${i}`,
      name: `GENAI_API_KEY_${i}`,
      status: 'inactive',
      configured: !!import.meta.env[envKey],
    } as any);
  }
  return keys;
}

const CircularLauncher: React.FC<CircularLauncherProps> = ({ onClick }) => {
  const [position, setPosition] = useState({ x: window.innerWidth - 100, y: 20 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [showApiKeySwap, setShowApiKeySwap] = useState(false);
  const [apiKeyOptions, setApiKeyOptions] = useState<ApiKeyOption[]>(getApiKeyOptions());
  const [currentApiKeyId, setCurrentApiKeyId] = useState(() => localStorage.getItem('selectedApiKeyId') || (apiKeyOptions[0]?.id || ''));
  const [isRestarting, setIsRestarting] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    setApiKeyOptions((opts) => opts.map(opt => ({ ...opt, status: opt.id === currentApiKeyId ? 'active' : 'inactive' })));
  }, [currentApiKeyId]);

  const [dragged, setDragged] = useState(false);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragged(false);
    setDragStart({
      x: e.clientX - position.x,
      y: e.clientY - position.y
    });
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (isDragging) {
      setDragged(true);
      const newX = e.clientX - dragStart.x;
      const newY = e.clientY - dragStart.y;
      const maxX = window.innerWidth - 80;
      const maxY = window.innerHeight - 80;
      setPosition({
        x: Math.max(0, Math.min(newX, maxX)),
        y: Math.max(0, Math.min(newY, maxY))
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, dragStart]);

  const handleClick = (e: React.MouseEvent) => {
    if (!dragged && !showApiKeySwap) {
      onClick();
    }
  };

  useEffect(() => {
    const handleResize = () => {
      const maxX = window.innerWidth - 80;
      const maxY = window.innerHeight - 80;
      setPosition(prev => ({
        x: Math.min(prev.x, maxX),
        y: Math.min(prev.y, maxY)
      }));
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleApiKeySwap = async (newApiKeyId: string) => {
    setIsRestarting(true);
    setCurrentApiKeyId(newApiKeyId);
    localStorage.setItem('selectedApiKeyId', newApiKeyId);
    setTimeout(() => {
      setShowApiKeySwap(false);
      setIsRestarting(false);
      window.location.reload();
    }, 1500);
  };

  const toggleApiKeySwap = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowApiKeySwap(!showApiKeySwap);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400 bg-green-400/20';
      case 'inactive': return 'text-gray-400 bg-gray-400/20';
      case 'error': return 'text-red-400 bg-red-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  if (isRestarting) {
    return (
      <div
        className="fixed w-20 h-20 bg-gradient-to-br from-confluence-blue to-confluence-light-blue text-white rounded-full shadow-2xl z-50 flex items-center justify-center font-bold text-sm backdrop-blur-xl border-2 border-white/30 animate-pulse"
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
        }}
      >
        <RotateCcw className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <>
      {/* Main Launcher Button */}
      <div className="fixed z-50" style={{ left: `${position.x}px`, top: `${position.y}px` }}>
        <div className="relative">
          <button
            ref={buttonRef}
            onMouseDown={handleMouseDown}
            onClick={handleClick}
            className="w-20 h-20 bg-gradient-to-br from-confluence-blue to-confluence-light-blue text-white rounded-full shadow-2xl cursor-move flex items-center justify-center font-bold text-sm backdrop-blur-xl border-2 border-white/30 hover:shadow-confluence-blue/50 hover:shadow-2xl transition-all duration-300"
            style={{
              background: `
                linear-gradient(135deg, 
                  rgba(38, 132, 255, 0.9) 0%, 
                  rgba(0, 82, 204, 0.9) 100%
                ),
                radial-gradient(circle at 30% 30%, 
                  rgba(255, 255, 255, 0.3) 0%, 
                  transparent 50%
                )
              `,
            }}
          >
            <span className="text-white font-extrabold tracking-tight">C.AIA</span>
          </button>

          {/* API Key Settings Button */}
          <button
            onClick={toggleApiKeySwap}
            className="absolute -bottom-2 -right-2 w-8 h-8 bg-white/90 backdrop-blur-xl text-confluence-blue rounded-full shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center border border-white/30"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* API Key Swap Panel */}
      {showApiKeySwap && (
        <div 
          className="fixed z-40"
          style={{ 
            left: `${Math.min(position.x + 100, window.innerWidth - 320)}px`, 
            top: `${position.y}px` 
          }}
        >
          <div className="bg-white/90 backdrop-blur-xl border border-white/30 rounded-2xl shadow-2xl w-80 overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-confluence-blue/90 to-confluence-light-blue/90 backdrop-blur-xl p-4 text-white border-b border-white/10">
              <div className="flex items-center space-x-3">
                <Key className="w-6 h-6" />
                <div>
                  <h3 className="text-lg font-bold">API Key Manager</h3>
                  <p className="text-blue-100/90 text-sm">Switch between available API keys</p>
                </div>
              </div>
            </div>

            {/* API Key Options */}
            <div className="p-4 space-y-3">
              {apiKeyOptions.every(opt => !opt.configured) && (
                <div className="text-red-500 text-sm text-center mb-2">
                  No API keys configured. Please add VITE_GENAI_API_KEY_1, VITE_GENAI_API_KEY_2, etc. to your .env file and restart the dev server.
                </div>
              )}
              {apiKeyOptions.map((option) => (
                <button
                  key={option.id}
                  onClick={() => option.configured && handleApiKeySwap(option.id)}
                  disabled={option.id === currentApiKeyId || !option.configured}
                  className={`w-full p-3 rounded-lg border transition-all duration-200 flex items-center justify-between ${
                    option.id === currentApiKeyId
                      ? 'bg-confluence-blue/20 border-confluence-blue/30 text-confluence-blue'
                      : option.configured
                        ? 'bg-white/60 border-white/30 hover:bg-white/80 text-gray-700'
                        : 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${getStatusColor(option.status)} ${!option.configured ? 'bg-gray-200' : ''}`} />
                    <div className="text-left">
                      <div className="font-medium">{option.name}</div>
                      <div className="text-xs opacity-70 capitalize">{option.configured ? option.status : 'not configured'}</div>
                    </div>
                  </div>
                  {option.id === currentApiKeyId && (
                    <Check className="w-5 h-5 text-confluence-blue" />
                  )}
                </button>
              ))}
            </div>

            {/* Footer */}
            <div className="p-4 bg-white/50 backdrop-blur-sm border-t border-white/20">
              <p className="text-xs text-gray-600 text-center">
                Switching API key will restart the launcher
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default CircularLauncher; 